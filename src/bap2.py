# src/bap2.py

import torch
import torch.nn as nn


class BAP2(nn.Module):
    """
    BAP2: Child-Anchor BAP con Heredity Gate contextual.

    Entrada esperada:
        encoder_hidden_states = [text_tokens | child_tokens | father_tokens | mother_tokens]

    Diferencia contra BAP1:
        BAP1:
            alpha = gate(father_tokens, mother_tokens)

        BAP2:
            alpha = gate(father_tokens, mother_tokens, A_father, A_mother)

    Fórmula:
        parent_mix = alpha * A_father + (1 - alpha) * A_mother

        out_face = child_scale * A_child
                 + parent_scale * parent_mix

    Nota:
        El gate debe operar en float32 para evitar NaN.
        En el attach BAP2 vamos a forzar:
            proc.gate.float()
    """

    def __init__(
        self,
        hidden_size: int,
        cross_attention_dim: int,
        num_tokens: int = 4,
        scale: float = 1.0,
        child_scale: float = 1.0,
        parent_scale: float = 0.1,
        gate_groups: int = 16,
        gate_hidden: int = 256,
        temperature: float = 1.0,
    ):
        super().__init__()

        if hidden_size % gate_groups != 0:
            raise ValueError(
                f"hidden_size={hidden_size} debe ser divisible entre "
                f"gate_groups={gate_groups}"
            )

        self.hidden_size = hidden_size
        self.cross_attention_dim = cross_attention_dim
        self.num_tokens = num_tokens

        self.scale = scale
        self.child_scale = child_scale
        self.parent_scale = parent_scale

        self.gate_groups = gate_groups
        self.temperature = temperature

        # Rama hijo
        self.to_k_child = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_child = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        # Rama padre
        self.to_k_father = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_father = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        # Rama madre
        self.to_k_mother = nn.Linear(cross_attention_dim, hidden_size, bias=False)
        self.to_v_mother = nn.Linear(cross_attention_dim, hidden_size, bias=False)

        # Gate contextual:
        # father_pool: cross_attention_dim
        # mother_pool: cross_attention_dim
        # A_father_pool: hidden_size
        # A_mother_pool: hidden_size
        gate_in_dim = 2 * cross_attention_dim + 2 * hidden_size

        # Gate contextual lineal:
        # alpha = sigmoid(W [father_tokens, mother_tokens, A_father, A_mother] + b)
        self.gate = nn.Linear(
            gate_in_dim,
            gate_groups,
            bias=True,
        )

        # Inicialización en cero:
        # sigmoid(0) = 0.5
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

        # Para inspección
        self.attn_map_child = None
        self.attn_map_father = None
        self.attn_map_mother = None
        self.last_alpha = None

    def _ip_attention(self, attn, query, key_states, value_states, which=None):
        key = attn.head_to_batch_dim(key_states)
        value = attn.head_to_batch_dim(value_states)

        probs = attn.get_attention_scores(query, key, None)

        if which == "child":
            self.attn_map_child = probs
        elif which == "father":
            self.attn_map_father = probs
        elif which == "mother":
            self.attn_map_mother = probs

        out = torch.bmm(probs, value)
        out = attn.batch_to_head_dim(out)

        return out

    def _expand_alpha(self, alpha, target_channels):
        """
        alpha:
            [B, gate_groups]

        salida:
            [B, 1, target_channels]
        """
        batch_size, groups = alpha.shape

        if target_channels % groups != 0:
            raise ValueError(
                f"target_channels={target_channels} no es divisible entre "
                f"gate_groups={groups}."
            )

        channels_per_group = target_channels // groups

        alpha = alpha.repeat_interleave(channels_per_group, dim=1)
        alpha = alpha.unsqueeze(1)

        return alpha

    def __call__(
        self,
        attn,
        hidden_states,
        encoder_hidden_states=None,
        attention_mask=None,
        temb=None,
        *args,
        **kwargs,
    ):
        residual = hidden_states

        if encoder_hidden_states is None:
            raise ValueError(
                "BAP2 requiere encoder_hidden_states con formato "
                "[text | child | father | mother]."
            )

        if attn.spatial_norm is not None:
            hidden_states = attn.spatial_norm(hidden_states, temb)

        input_ndim = hidden_states.ndim

        if input_ndim == 4:
            batch_size, channel, height, width = hidden_states.shape
            hidden_states = hidden_states.view(
                batch_size,
                channel,
                height * width,
            ).transpose(1, 2)

        batch_size = hidden_states.shape[0]

        total_tokens = encoder_hidden_states.shape[1]
        t = self.num_tokens
        text_len = total_tokens - 3 * t

        if text_len <= 0:
            raise ValueError(
                f"encoder_hidden_states inválido: total_tokens={total_tokens}, "
                f"num_tokens={t}. Se esperaba [text | child | father | mother]."
            )

        text_states = encoder_hidden_states[:, :text_len, :]
        child_states = encoder_hidden_states[:, text_len:text_len + t, :]
        father_states = encoder_hidden_states[:, text_len + t:text_len + 2 * t, :]
        mother_states = encoder_hidden_states[:, text_len + 2 * t:text_len + 3 * t, :]

        sequence_length = text_states.shape[1]

        attention_mask = attn.prepare_attention_mask(
            attention_mask,
            sequence_length,
            batch_size,
        )

        if attn.group_norm is not None:
            hidden_states = attn.group_norm(
                hidden_states.transpose(1, 2)
            ).transpose(1, 2)

        # Query
        query = attn.to_q(hidden_states)
        query = attn.head_to_batch_dim(query)

        # Atención textual estándar
        if attn.norm_cross:
            text_states = attn.norm_encoder_hidden_states(text_states)

        key_text = attn.to_k(text_states)
        value_text = attn.to_v(text_states)

        key_text = attn.head_to_batch_dim(key_text)
        value_text = attn.head_to_batch_dim(value_text)

        attention_probs = attn.get_attention_scores(
            query,
            key_text,
            attention_mask,
        )

        hidden_states = torch.bmm(attention_probs, value_text)
        hidden_states = attn.batch_to_head_dim(hidden_states)

        # Rama hijo
        k_child = self.to_k_child(child_states)
        v_child = self.to_v_child(child_states)

        A_child = self._ip_attention(
            attn,
            query,
            k_child,
            v_child,
            which="child",
        )

        # Rama padre
        k_father = self.to_k_father(father_states)
        v_father = self.to_v_father(father_states)

        A_father = self._ip_attention(
            attn,
            query,
            k_father,
            v_father,
            which="father",
        )

        # Rama madre
        k_mother = self.to_k_mother(mother_states)
        v_mother = self.to_v_mother(mother_states)

        A_mother = self._ip_attention(
            attn,
            query,
            k_mother,
            v_mother,
            which="mother",
        )

        # ====================================================
        # Heredity Gate contextual V2 en float32
        # ====================================================

        father_pool = father_states.mean(dim=1)   # [B, cross_attention_dim]
        mother_pool = mother_states.mean(dim=1)   # [B, cross_attention_dim]

        A_father_pool = A_father.mean(dim=1)      # [B, hidden_size]
        A_mother_pool = A_mother.mean(dim=1)      # [B, hidden_size]

        gate_input = torch.cat(
            [
                father_pool.float(),
                mother_pool.float(),
                A_father_pool.float(),
                A_mother_pool.float(),
            ],
            dim=1,
        )

        logits = self.gate(gate_input)

        if self.temperature != 1.0:
            logits = logits / self.temperature

        alpha = torch.sigmoid(logits)             # [B, gate_groups]

        self.last_alpha = alpha.detach()

        alpha_expanded = self._expand_alpha(
            alpha,
            target_channels=A_father.shape[-1],
        )

        # Volver al dtype de las atenciones para mezclar sin conflicto
        alpha_expanded = alpha_expanded.to(dtype=A_father.dtype)

        parent_mix = (
            alpha_expanded * A_father
            + (1.0 - alpha_expanded) * A_mother
        )

        out_face = (
            self.child_scale * A_child
            + self.parent_scale * parent_mix
        )

        hidden_states = hidden_states + self.scale * out_face

        # Salida estándar
        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)

        if input_ndim == 4:
            hidden_states = hidden_states.transpose(-1, -2).reshape(
                batch_size,
                channel,
                height,
                width,
            )

        if attn.residual_connection:
            hidden_states = hidden_states + residual

        hidden_states = hidden_states / attn.rescale_output_factor

        return hidden_states