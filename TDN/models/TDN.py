import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()

        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.enc_in = configs.enc_in
        self.model_type = configs.model_type
        self.d_model = configs.d_model
        self.use_revin = configs.use_revin
        
        self.cluster = 5
        self.mem_dim = 12 # self.pred_len
        self.mem_dim2 = 24 # self.pred_len
        self.mem_dim3 = 48 # self.pred_len
        self.mem_dim4 = self.pred_len
        self.period_len = configs.period_len

        # Memory
        self.time_emb = nn.Parameter(
            torch.zeros(self.period_len, self.cluster, self.mem_dim, self.enc_in))
        self.time_emb2 = nn.Parameter(
            torch.zeros(self.period_len, self.cluster, self.mem_dim2, self.enc_in))
        self.time_emb3 = nn.Parameter(
            torch.zeros(self.period_len, self.cluster, self.mem_dim3, self.enc_in))
        self.time_emb4 = nn.Parameter(
            torch.zeros(self.period_len, self.cluster, self.mem_dim4, self.enc_in))
        # nn.init.xavier_uniform_(self.time_emb)
        # self.time_emb = self.time_emb*0.2


        self.weights = nn.ModuleList()
        self.prox_linears = nn.ModuleList()
        self.weights2 = nn.ModuleList()
        self.prox_linears2 = nn.ModuleList()
        self.weights3 = nn.ModuleList()
        self.prox_linears3 = nn.ModuleList()
        self.weights4 = nn.ModuleList()
        self.prox_linears4 = nn.ModuleList()

        self.time_len = int(self.pred_len / self.mem_dim)
        self.time_len2 = int(self.pred_len / self.mem_dim2)
        self.time_len3 = int(self.pred_len / self.mem_dim3)
        self.time_len4 = int(self.pred_len / self.mem_dim4)

        self.balance = nn.Parameter(torch.ones(self.period_len, self.time_len, self.enc_in, 2)/2)
        self.balance2 = nn.Parameter(torch.ones(self.period_len, self.time_len2, self.enc_in, 2)/2)
        self.balance3 = nn.Parameter(torch.ones(self.period_len, self.time_len3, self.enc_in, 2)/2)
        self.balance4 = nn.Parameter(torch.ones(self.period_len, self.time_len4, self.enc_in, 2)/2)
        for i in range(self.time_len):
            self.weights.append(nn.Linear(self.seq_len, self.cluster))
            self.prox_linears.append(nn.Linear(self.seq_len, self.mem_dim))
            self.weights2.append(nn.Linear(self.seq_len, self.cluster))
            self.prox_linears2.append(nn.Linear(self.seq_len, self.mem_dim2))
            self.weights3.append(nn.Linear(self.seq_len, self.cluster))
            self.prox_linears3.append(nn.Linear(self.seq_len, self.mem_dim3))
            self.weights4.append(nn.Linear(self.seq_len, self.cluster))
            self.prox_linears4.append(nn.Linear(self.seq_len, self.mem_dim4))
        # self.prox_linear = nn.Linear(self.seq_len, self.pred_len)
        # self.weight = nn.Linear(self.seq_len, self.cluster)

    def forward(self, x, time_stamp):

        # instance norm
        if self.use_revin:
            seq_mean = torch.mean(x, dim=1, keepdim=True)
            seq_var = torch.var(x, dim=1, keepdim=True) + 1e-5
            x = (x - seq_mean) / torch.sqrt(seq_var)

        # print(torch.sum(self.balance, dim=-1).shape, self.balance.shape)
        # balance = self.balance / torch.sum(self.balance, dim=-1).unsqueeze(-1)
        balance2 = self.balance2 / torch.sum(self.balance2, dim=-1).unsqueeze(-1)
        balance3 = self.balance3 / torch.sum(self.balance3, dim=-1).unsqueeze(-1)
        balance4 = self.balance4 / torch.sum(self.balance4, dim=-1).unsqueeze(-1)

        # period_ori = self.time_emb[time_stamp].permute(0, 1, 3, 2)
        period_ori2 = self.time_emb2[time_stamp].permute(0, 1, 3, 2)
        period_ori3 = self.time_emb3[time_stamp].permute(0, 1, 3, 2)
        period_ori4 = self.time_emb4[time_stamp].permute(0, 1, 3, 2)

        # 3
        # seg_list = []
        # for i in range(self.time_len):
        #     weight = torch.nn.functional.softmax(self.weights[i](x.permute(0, 2, 1)).permute(0, 2, 1), dim=1)
        #     period = torch.sum(torch.einsum('ijkl, ijk->ijkl', period_ori, weight), dim=1) # 256 5 7 96, 256 5 7 -> 256 7 96
        #     prox = self.prox_linears[i](x.permute(0, 2, 1))
        #     pre_sum = balance[time_stamp, i, :, 0].unsqueeze(-1) * prox + balance[time_stamp, i, :, 1].unsqueeze(-1) * period
        #     # pre_sum = 0*prox + 1*period
        #     seg_list.append(pre_sum.permute(0, 2, 1))
        # period1 = torch.cat(seg_list, dim = 1)

        seg_list2 = []
        for i in range(self.time_len2):
            weight = torch.nn.functional.softmax(self.weights2[i](x.permute(0, 2, 1)).permute(0, 2, 1), dim=1)
            period = torch.sum(torch.einsum('ijkl, ijk->ijkl', period_ori2, weight), dim=1) # 256 5 7 96, 256 5 7 -> 256 7 96
            p2 = period
            prox = self.prox_linears2[i](x.permute(0, 2, 1))
            pre_sum = balance2[time_stamp, i, :, 0].unsqueeze(-1) * prox + balance2[time_stamp, i, :, 1].unsqueeze(-1) * period
            # pre_sum = 0*prox + 1*period
            seg_list2.append(pre_sum.permute(0, 2, 1))
        period2 = torch.cat(seg_list2, dim = 1)

        seg_list3 = []
        for i in range(self.time_len3):
            weight = torch.nn.functional.softmax(self.weights3[i](x.permute(0, 2, 1)).permute(0, 2, 1), dim=1)
            period = torch.sum(torch.einsum('ijkl, ijk->ijkl', period_ori3, weight), dim=1) # 256 5 7 96, 256 5 7 -> 256 7 96
            prox = self.prox_linears3[i](x.permute(0, 2, 1))
            pre_sum = balance3[time_stamp, i, :, 0].unsqueeze(-1) * prox + balance3[time_stamp, i, :, 1].unsqueeze(-1) * period
            # pre_sum = 0*prox + 1*period
            seg_list3.append(pre_sum.permute(0, 2, 1))
        period3 = torch.cat(seg_list3, dim = 1)

        seg_list4 = []
        for i in range(self.time_len4):
            weight = torch.nn.functional.softmax(self.weights4[i](x.permute(0, 2, 1)).permute(0, 2, 1), dim=1)
            period = torch.sum(torch.einsum('ijkl, ijk->ijkl', period_ori4, weight), dim=1) # 256 5 7 96, 256 5 7 -> 256 7 96
            prox = self.prox_linears4[i](x.permute(0, 2, 1))
            pre_sum = balance4[time_stamp, i, :, 0].unsqueeze(-1) * prox + balance4[time_stamp, i, :, 1].unsqueeze(-1) * period
            # pre_sum = 0*prox + 1*period
            seg_list4.append(pre_sum.permute(0, 2, 1))
        period4 = torch.cat(seg_list4, dim = 1)

        prediction = (period2 + period3 + period4) / 3
        
        # instance denorm
        if self.use_revin:
            prediction = prediction * torch.sqrt(seq_var) + seq_mean
        
        return prediction, self.time_emb2, p2