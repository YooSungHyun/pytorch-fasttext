import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):
    def __init__(self, vocab_size, embedding_size):
        super(Net, self).__init__()
        self.embeddings = nn.Embedding(vocab_size, embedding_size)
        self.out_embeddings = nn.Embedding(vocab_size, embedding_size)

    def forward(self, input_ids, labels, negative_samples):
        # 문맥 단어들의 임베딩을 조회
        # frequency sampling 대신 드롭아웃으로 그냥 운좋게 비슷한 효과를 낼 수 있지 않을까?
        input_pad_mask = input_ids != 0
        input_pad_mask = input_pad_mask.float()
        context_embeds = self.embeddings(input_ids)
        input_pad_mask = input_pad_mask.unsqueeze(-1).expand_as(context_embeds)
        masked_embeddings = context_embeds * input_pad_mask

        v_t = masked_embeddings.sum(dim=1)
        # 타겟 단어의 임베딩 조회
        target_embeds = self.out_embeddings(labels)
        # 네거티브 샘플의 임베딩 조회
        negative_embeds = self.out_embeddings(negative_samples)

        # 3개의 단어를 평균함 == 평균은 동등 == 즉 모든 feature에 동등한 기회를 주겠다
        target_embeds_mean = torch.mean(target_embeds, dim=1).unsqueeze(1)
        # 타겟 단어에 대한 유사도 관계 수치화
        positive_score = torch.bmm(target_embeds_mean, v_t.unsqueeze(2)).squeeze()
        # 네거티브 샘플에 대한 유사도 관계 수치화
        negative_score = torch.bmm(negative_embeds, v_t.unsqueeze(2)).squeeze()
        return positive_score, negative_score

    def predict(self, input_ids):
        context_embeds = self.embeddings(input_ids)
        return context_embeds
