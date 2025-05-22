import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os, sys
import math
import pandas as pd


from transformers import RobertaTokenizer, RobertaModel, TimesformerModel, Data2VecAudioModel

class Text_model(nn.Module):
    def __init__(self, text_model, clsNum):
        super(Text_model, self).__init__()

        """Text Model"""
        tmodel_path = os.path.join("../pretrained_model", text_model)
        if text_model == "roberta-large":
            self.text_model = RobertaModel.from_pretrained(tmodel_path)
            tokenizer = RobertaTokenizer.from_pretrained(tmodel_path)
            self.speaker_list = ['<s1>', '<s2>', '<s3>', '<s4>', '<s5>', '<s6>', '<s7>', '<s8>', '<s9>']
            self.speaker_tokens_dict = {'additional_special_tokens': self.speaker_list}
            tokenizer.add_special_tokens(self.speaker_tokens_dict)
        self.text_model.resize_token_embeddings(len(tokenizer))
        self.text_hidden_dim = self.text_model.config.hidden_size

        """Logit"""
        self.W = nn.Linear(self.text_hidden_dim, 768)
        self.classifier = nn.Linear(768, clsNum)

    def forward(self, batch_text_tokens, attention_masks):
        batch_context_output = self.text_model(input_ids=batch_text_tokens, attention_mask=attention_masks).last_hidden_state[:,-1,:] # (batch, 1024)
        batch_last_hidden = self.W(batch_context_output)
        context_logit = self.classifier(batch_last_hidden)

        return batch_last_hidden, context_logit


class Audio_model(nn.Module):
    def __init__(self, audio_model, clsNum, init_config):
        super(Audio_model, self).__init__()

        """Audio Model"""
        amodel_path = os.path.join("../pretrained_model", audio_model)
        if audio_model == "data2vec-audio-base-960h":

            self.model = Data2VecAudioModel.from_pretrained(amodel_path)
            self.model.config.update(init_config.__dict__)

        self.hiddenDim = self.model.config.hidden_size

        """score"""
        self.W = nn.Linear(self.hiddenDim, clsNum)

    def forward(self, batch_input):

        batch_audio_output = self.model(batch_input).last_hidden_state[:,0,:] # (batch, 768)
        audio_logit = self.W(batch_audio_output) # (batch, clsNum)

        return batch_audio_output, audio_logit

class Video_model(nn.Module):
    def __init__(self, video_model, clsNum):
        super(Video_model, self).__init__()

        """Video Model"""
        vmodel_path  = os.path.join('../pretrained_model/',video_model)
        if video_model == "timesformer-base-finetuned-k400":
            self.model = TimesformerModel.from_pretrained(vmodel_path)
        self.hiddenDim = self.model.config.hidden_size

        """score"""
        self.W = nn.Linear(self.hiddenDim, clsNum)

    def forward(self, batch_input):

        batch_video_output = self.model(batch_input).last_hidden_state[:,0,:] # (batch, 768)
        video_logit = self.W(batch_video_output) # (batch, clsNum)

        return batch_video_output, video_logit


class Audio_model_emotion2vec(nn.Module):
    def __init__(self, hidden_dim, clsNum):
        super(Audio_model_emotion2vec, self).__init__()

        self.proj = nn.Linear(768, hidden_dim)
        self.W = nn.Linear(hidden_dim, clsNum)

    def forward(self, batch_input):
        batch_audio_output = F.relu(self.proj(batch_input))
        audio_logit = self.W(batch_audio_output)
        return batch_audio_output, audio_logit


class Fusion_model(nn.Module):
    def __init__(self, args, clsNum):
        super(Fusion_model, self).__init__()

        self.input_dim = 768
        self.hidden_dim = 768
        self.dropout_prob = 0.1

        self.text_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.audio_proj = nn.Linear(self.input_dim, self.hidden_dim)
        self.video_proj = nn.Linear(self.input_dim, self.hidden_dim)

        self.LN_t = nn.LayerNorm(self.hidden_dim)
        self.LN_a = nn.LayerNorm(self.hidden_dim)
        self.LN_v = nn.LayerNorm(self.hidden_dim)

        self.DP_t = nn.Dropout(self.dropout_prob)
        self.DP_a = nn.Dropout(self.dropout_prob)
        self.DP_v = nn.Dropout(self.dropout_prob)

        # 共享参数
        self.share = nn.Sequential(
            nn.Linear(self.hidden_dim, 3*self.hidden_dim),
            nn.ReLU(self.dropout_prob),
            nn.Linear(3*self.hidden_dim, self.hidden_dim),
        )

        self.output_proj = nn.Linear(3 * self.hidden_dim, self.hidden_dim)
        self.W = nn.Linear(self.hidden_dim, clsNum)


    def forward(self, text_emb, audio_emb, video_emb):
        text_emb = self.text_proj(text_emb)
        audio_emb = self.audio_proj(audio_emb)
        video_emb = self.video_proj(video_emb)

        text_emb = self.LN_t(text_emb)
        audio_emb = self.LN_a(audio_emb)
        video_emb = self.LN_v(video_emb)

        text_emb = self.DP_t(text_emb)
        audio_emb = self.DP_a(audio_emb)
        video_emb = self.DP_v(video_emb)

        text_emb = self.share(text_emb)
        audio_emb = self.share(audio_emb)
        video_emb = self.share(video_emb)

        fused_features = F.relu(self.output_proj(torch.cat([text_emb, audio_emb, video_emb], dim=-1)))
        logit = self.W(fused_features)
        return fused_features, logit.transpose(0, 1)


class ASF(nn.Module):
    def __init__(self, clsNum, hidden_size, beta_shift, dropout_prob, num_head):
        super(ASF, self).__init__()

        self.TEXT_DIM = 768
        self.VISUAL_DIM = 768
        self.ACOUSTIC_DIM = 768
        self.multihead_attn = nn.MultiheadAttention(self.VISUAL_DIM + self.ACOUSTIC_DIM, num_head)

        self.W_hav = nn.Linear(self.VISUAL_DIM + self.ACOUSTIC_DIM + self.TEXT_DIM, self.TEXT_DIM)

        self.W_av = nn.Linear(self.VISUAL_DIM + self.ACOUSTIC_DIM, self.TEXT_DIM)

        self.beta_shift = beta_shift

        self.LayerNorm = nn.LayerNorm(hidden_size)
        self.AV_LayerNorm = nn.LayerNorm(self.VISUAL_DIM + self.ACOUSTIC_DIM)
        self.dropout = nn.Dropout(dropout_prob)

        """Logit"""
        self.W = nn.Linear(self.TEXT_DIM, clsNum)

    def forward(self, text_embedding, visual, acoustic):
        eps = 1e-6
        nv_embedd = torch.cat((visual, acoustic), dim=-1)
        new_nv = self.multihead_attn(nv_embedd, nv_embedd, nv_embedd)[0] + nv_embedd

        av_embedd = self.dropout(self.AV_LayerNorm(new_nv))

        weight_av = F.relu(self.W_hav(torch.cat((av_embedd, text_embedding), dim=-1)))

        h_m = weight_av * self.W_av(av_embedd)

        em_norm = text_embedding.norm(2, dim=-1)
        hm_norm = h_m.norm(2, dim=-1)

        hm_norm_ones = torch.ones(hm_norm.shape, requires_grad=True).cuda()
        hm_norm = torch.where(hm_norm == 0, hm_norm_ones, hm_norm)

        thresh_hold = (em_norm / (hm_norm + eps)) * self.beta_shift

        ones = torch.ones(thresh_hold.shape, requires_grad=True).cuda()

        alpha = torch.min(thresh_hold, ones)
        alpha = alpha.unsqueeze(dim=-1)

        acoustic_vis_embedding = alpha * h_m

        embedding_output = self.dropout(
            self.LayerNorm(acoustic_vis_embedding + text_embedding)
        )

        logits = self.W(embedding_output)
        return logits



class CLModel(nn.Module):
    def __init__(self, args, clsNum, input_dim, hidden_dim, attention_heads=8, dropout=0.1):
        super(CLModel, self).__init__()
        self.args = args

        self.text_proj = nn.Linear(input_dim, hidden_dim)
        self.audio_proj = nn.Linear(input_dim, hidden_dim)
        self.video_proj = nn.Linear(input_dim, hidden_dim)

        self.attention = nn.MultiheadAttention(hidden_dim, attention_heads, dropout=dropout)
        self.proj = nn.Linear(3 * hidden_dim, hidden_dim)
        self.W = nn.Linear(hidden_dim, clsNum)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def get_reps(self, text_emb, audio_emb, video_emb):
        text_proj = self.text_proj(text_emb)        # (batch_size, output_dim)
        audio_proj = self.audio_proj(audio_emb)
        video_proj = self.video_proj(video_emb)

        stacked_features = torch.stack([text_proj, audio_proj, video_proj], dim=1)       # (batch_size, 3, output_dim)
        attn_output, _ = self.attention(stacked_features, stacked_features, stacked_features)    # (batch_size, 3, output_dim)

        attn_output += stacked_features

        attn_output = attn_output.reshape(attn_output.shape[0], -1)
        fused_features = self.proj(self.dropout(self.activation(attn_output)))
        logit = self.W(fused_features)

        return logit, fused_features

    def forward(self, reps, centers, score_func):
        num_classes, num_centers = centers.shape[0], centers.shape[1]
        reps = reps.unsqueeze(1).expand(reps.shape[0], num_centers, -1)
        reps = reps.unsqueeze(1).expand(reps.shape[0], num_classes, num_centers, -1)

        centers = centers.unsqueeze(0).expand(reps.shape[0], -1, -1, -1)
        # batch * turn, num_classes, num_centers
        sim_matrix = score_func(reps, centers)

        # batch * turn, num_calsses
        scores = sim_matrix
        return scores


class PositionalEncoding(nn.Module):
    def __init__(self, dim, max_len=512):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_len, dim)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, dim, 2) * (-math.log(10000.0) / dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x, speaker_emb):
        L = x.size(1)
        pos_emb = self.pe[:, :L]
        x = x + pos_emb + speaker_emb
        # x = x + pos_emb
        return x


class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()

        assert d_model % n_heads == 0
        self.dim_per_head = d_model // n_heads
        self.d_model = d_model
        self.n_heads = n_heads

        self.proj_q = nn.Linear(d_model, n_heads * self.dim_per_head)
        self.proj_k = nn.Linear(d_model, n_heads * self.dim_per_head)
        self.proj_v = nn.Linear(d_model, n_heads * self.dim_per_head)

        nn.init.xavier_uniform_(self.proj_q.weight)
        nn.init.xavier_uniform_(self.proj_k.weight)
        nn.init.xavier_uniform_(self.proj_v.weight)
        nn.init.constant_(self.proj_q.bias, 0)
        nn.init.constant_(self.proj_k.bias, 0)
        nn.init.constant_(self.proj_v.bias, 0)

        self.dropout = nn.Dropout(dropout)
        self.softmax = nn.Softmax(dim=-1)

        self.relu = nn.ReLU()

        self.fc = nn.Linear(n_heads * self.dim_per_head, d_model)

    def forward(self, query, key, value, mask=None, type='self'):
        batch_size = query.size(0)
        dim_per_head = self.dim_per_head
        n_heads = self.n_heads

        query = self.proj_q(query).view(batch_size, -1, n_heads, dim_per_head).transpose(1, 2)
        key = self.proj_k(key).view(batch_size, -1, n_heads, dim_per_head).transpose(1, 2)
        value = self.proj_v(value).view(batch_size, -1, n_heads, dim_per_head).transpose(1, 2)

        query = query / math.sqrt(dim_per_head)
        scores = torch.matmul(query, key.transpose(2, 3))

        if mask is not None:
            mask = mask.unsqueeze(1).expand_as(scores)
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.softmax(self.relu(scores))
        drop_attn = self.dropout(attn)

        result = torch.matmul(drop_attn, value).transpose(1, 2).contiguous().view(batch_size, -1, n_heads * dim_per_head)
        output = self.fc(result)
        return output


class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.activation = nn.GELU()

    def forward(self, x):
        inter = self.dropout1(self.activation(self.w_1(self.layer_norm(x))))
        output = self.dropout2(self.w_2(inter))
        return output + x


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, n_head, dropout):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = MultiHeadedAttention(n_head, d_model, dropout=dropout)
        self.feed_forward = PositionwiseFeedForward(d_model, d_ff, dropout)
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, iter, inputs_a, inputs_b, mask):
        if inputs_a.equal(inputs_b):
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b
            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_b, inputs_b, mask=mask, type='self')
        else:
            if (iter != 0):
                inputs_b = self.layer_norm(inputs_b)
            else:
                inputs_b = inputs_b

            mask = mask.unsqueeze(1)
            context = self.self_attn(inputs_b, inputs_a, inputs_a, mask=mask, type='cross')

        out = self.dropout(context) + inputs_b
        return self.feed_forward(out)



class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_ff, n_head, num_layers, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.layers = num_layers
        self.pos_emb = PositionalEncoding(d_model)
        self.transformer_inter = nn.ModuleList(
            [TransformerEncoderLayer(d_model, d_ff, n_head, dropout) for _ in range(self.layers)]
        )
        self.dropout = nn.Dropout(dropout)

        # self.LN_a = nn.LayerNorm(d_model)
        # self.LN_b = nn.LayerNorm(d_model)

    def forward(self, x_a, x_b, mask, speaker_emb):
        if x_a.equal(x_b):
            x_b = self.pos_emb(x_b, speaker_emb)
            for i in range(self.layers):
                x_b = self.transformer_inter[i](i, x_b, x_b, mask.eq(0))
        else:
            x_a = self.pos_emb(x_a, speaker_emb)
            x_a = self.dropout(x_a)
            x_b = self.pos_emb(x_b, speaker_emb)
            x_b = self.dropout(x_b)
            for i in range(self.layers):
                # x_a = self.LN_a(x_a)
                # x_b = self.LN_b(x_b)
                x_b = self.transformer_inter[i](i, x_a, x_b, mask.eq(0))
        return x_b


class Unimodal_GatedFusion(nn.Module):
    def __init__(self, hidden_dim):
        super(Unimodal_GatedFusion, self).__init__()
        self.fc = nn.Linear(hidden_dim, hidden_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        w = torch.sigmoid(self.relu(self.fc(x)))
        return w * x


class Transformer_Based_Model(nn.Module):
    def __init__(self, args):
        super(Transformer_Based_Model, self).__init__()

        self.temp = args.temp
        self.clsNum = args.clsNum
        self.n_speaker = 304
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.speaker_embeddings = nn.Embedding(self.n_speaker + 1, args.hidden_dim, padding_idx=2).to(self.device)
        nn.init.normal_(self.speaker_embeddings.weight, mean=0, std=0.1)

        # Intra- and Inter-model Transformers
        self.t_t = TransformerEncoder(d_model=args.hidden_dim, d_ff=args.hidden_dim, n_head=args.n_head, num_layers=1, dropout=args.dropout)
        self.a_t = TransformerEncoder(d_model=args.hidden_dim, d_ff=args.hidden_dim, n_head=args.n_head, num_layers=1, dropout=args.dropout)
        self.v_t = TransformerEncoder(d_model=args.hidden_dim, d_ff=args.hidden_dim, n_head=args.n_head, num_layers=1, dropout=args.dropout)

        self.a_a = TransformerEncoder(d_model=args.hidden_dim, d_ff=args.hidden_dim, n_head=args.n_head, num_layers=1, dropout=args.dropout)
        self.t_a = TransformerEncoder(d_model=args.hidden_dim, d_ff=args.hidden_dim, n_head=args.n_head, num_layers=1, dropout=args.dropout)
        self.v_a = TransformerEncoder(d_model=args.hidden_dim, d_ff=args.hidden_dim, n_head=args.n_head, num_layers=1, dropout=args.dropout)

        self.v_v = TransformerEncoder(d_model=args.hidden_dim, d_ff=args.hidden_dim, n_head=args.n_head, num_layers=1, dropout=args.dropout)
        self.t_v = TransformerEncoder(d_model=args.hidden_dim, d_ff=args.hidden_dim, n_head=args.n_head, num_layers=1, dropout=args.dropout)
        self.a_v = TransformerEncoder(d_model=args.hidden_dim, d_ff=args.hidden_dim, n_head=args.n_head, num_layers=1, dropout=args.dropout)

        # Unimodal-level Gated Fusion
        self.t_t_gate = Unimodal_GatedFusion(args.hidden_dim)
        self.a_t_gate = Unimodal_GatedFusion(args.hidden_dim)
        self.v_t_gate = Unimodal_GatedFusion(args.hidden_dim)

        self.a_a_gate = Unimodal_GatedFusion(args.hidden_dim)
        self.t_a_gate = Unimodal_GatedFusion(args.hidden_dim)
        self.v_a_gate = Unimodal_GatedFusion(args.hidden_dim)

        self.v_v_gate = Unimodal_GatedFusion(args.hidden_dim)
        self.t_v_gate = Unimodal_GatedFusion(args.hidden_dim)
        self.a_v_gate = Unimodal_GatedFusion(args.hidden_dim)

        self.features_reduce_t = nn.Linear(3 * args.hidden_dim, args.hidden_dim)
        self.features_reduce_a = nn.Linear(3 * args.hidden_dim, args.hidden_dim)
        self.features_reduce_v = nn.Linear(3 * args.hidden_dim, args.hidden_dim)

        # Emotion Classifier
        self.fc = nn.Linear(args.hidden_dim*3, args.hidden_dim)
        self.w_t = nn.Linear(args.hidden_dim, self.clsNum)
        self.w_a = nn.Linear(args.hidden_dim, self.clsNum)
        self.w_v = nn.Linear(args.hidden_dim, self.clsNum)

    def forward(self, text, audio, video, u_mask, q_mask, dia_len):  # text:(sql, batch, hidden) umask:(batch, sql)
        spk_idx = q_mask
        origin_spk_idx = spk_idx
        for i, x in enumerate(dia_len):
            spk_idx[i, x:] = (2 * torch.ones(origin_spk_idx[i].size(0) - x)).int().cuda()
        spk_idx = spk_idx.long().cuda()  # 确保转换为 long 类型
        spk_embeddings = self.speaker_embeddings(spk_idx)

        text = text.transpose(0, 1)
        audio = audio.transpose(0, 1)
        video = video.transpose(0, 1)


        a_a_transformer_out = self.a_a(audio, audio, u_mask, spk_embeddings)
        t_a_transformer_out = self.t_a(text, audio, u_mask, spk_embeddings)
        v_a_transformer_out = self.v_a(video, audio, u_mask, spk_embeddings)
        a_a_transformer_out = self.a_a_gate(a_a_transformer_out)
        t_a_transformer_out = self.t_a_gate(t_a_transformer_out)
        v_a_transformer_out = self.v_a_gate(v_a_transformer_out)
        a_transformer_out = self.features_reduce_a(torch.cat([a_a_transformer_out, t_a_transformer_out, v_a_transformer_out], dim=-1))


        v_v_transformer_out = self.v_v(video, video, u_mask, spk_embeddings)
        t_v_transformer_out = self.t_v(text, video, u_mask, spk_embeddings)
        a_v_transformer_out = self.a_v(audio, video, u_mask, spk_embeddings)
        v_v_transformer_out = self.v_v_gate(v_v_transformer_out)
        t_v_transformer_out = self.t_v_gate(t_v_transformer_out)
        a_v_transformer_out = self.a_v_gate(a_v_transformer_out)
        v_transformer_out = self.features_reduce_v(torch.cat([v_v_transformer_out, t_v_transformer_out, a_v_transformer_out], dim=-1))


        t_t_transformer_out = self.t_t(text, text, u_mask, spk_embeddings)
        a_t_transformer_out = self.t_t(a_transformer_out, text, u_mask, spk_embeddings)
        v_t_transformer_out = self.t_t(v_transformer_out, text, u_mask, spk_embeddings)
        t_t_transformer_out = self.t_t_gate(t_t_transformer_out)
        a_t_transformer_out = self.t_t_gate(a_t_transformer_out)
        v_t_transformer_out = self.t_t_gate(v_t_transformer_out)

        final_transformer_out = self.features_reduce_t(torch.cat([t_t_transformer_out, a_t_transformer_out, v_t_transformer_out], dim=-1))


        # return t_log_probs, a_log_probs, v_log_probs, all_log_probs, all_probs, kl_t_log_prob, kl_a_log_prob, kl_v_log_prob, kl_all_prob
        # hidden = self.fc(torch.cat([t_t_transformer_out, a_t_transformer_out, v_t_transformer_out], dim=-1))

        t_logits = self.w_t(final_transformer_out)
        a_logits = self.w_a(a_transformer_out)
        v_logits = self.w_v(v_transformer_out)
        return t_logits, a_logits, v_logits, final_transformer_out, a_transformer_out, v_transformer_out


class MaskedKLDivLoss(nn.Module):
    def __init__(self):
        super(MaskedKLDivLoss, self).__init__()
        self.loss = nn.KLDivLoss(reduction='sum')

    def forward(self, log_pred, target, mask):
        mask_ = mask.view(-1, 1)
        loss = self.loss(log_pred * mask_, target * mask_) / torch.sum(mask)
        return loss

class MaskedNLLLoss(nn.Module):
    def __init__(self, weight=None):
        super(MaskedNLLLoss, self).__init__()
        self.weight = weight
        self.loss = nn.NLLLoss(weight=weight, reduction='sum')

    def forward(self, pred, target, mask):
        mask_ = mask.view(-1, 1)
        if type(self.weight) == type(None):
            loss = self.loss(pred * mask_, target) / torch.sum(mask)
        else:
            loss = self.loss(pred * mask_, target) \
                   / torch.sum(self.weight[target] * mask_.squeeze())
        return loss


class TestModel(nn.Module):
    def __init__(self, args, clsNum=6):
        super(TestModel, self).__init__()
        self.clsNum = clsNum
        self.hidden_dim = args.hidden_dim

        self.gate_t = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_a = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.gate_v = nn.Linear(self.hidden_dim, self.hidden_dim)

        self.fc = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.w = nn.Linear(self.hidden_dim, self.hidden_dim)

    def forward(self, text, audio, video):
        t_gate = torch.sigmoid(self.gate_t(text))
        a_gate = torch.sigmoid(self.gate_a(audio))
        v_gate = torch.sigmoid(self.gate_v(video))

        text = text * t_gate
        audio = audio * a_gate
        video = video * v_gate

        combine = self.fc(torch.cat([text, audio, video], dim=-1))

        logits = self.w(combine)

        return logits







