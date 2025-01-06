import torch
from torch import nn
class LambdaRankLoss(nn.Module):
    def __init__(self, k=10):
        """
        k: The rank position up to which we compute the NDCG.
        """
        super(LambdaRankLoss, self).__init__()
        self.k = k

    def dcg_at_k(self, relevance_scores, k):
        """
        Compute DCG@k.
        """
        relevance_scores = relevance_scores[:k]
        gains = torch.pow(2, relevance_scores) - 1
        discounts = torch.log2(torch.arange(len(relevance_scores), dtype=torch.float32).to("cuda") + 2)
        
        return torch.sum(gains / discounts)

    def ndcg_at_k(self, true_relevance, predicted_scores, k):
        """
        Compute NDCG@k.
        """

        _, indices = torch.sort(predicted_scores, descending=True)
        
        
        sorted_true_relevance = true_relevance[indices]
        
        
        ideal_sorted_relevance = torch.sort(true_relevance, descending=True)[0]

        dcg = self.dcg_at_k(sorted_true_relevance, k)
        ideal_dcg = self.dcg_at_k(ideal_sorted_relevance, k)

        return 1-(dcg / ideal_dcg) if ideal_dcg > 0 else torch.tensor(0.0)
    

    def forward_(self,y_pred,true_labels):

        y_true = (true_labels-true_labels.min())

        if y_true.max()> 0:
            y_true = y_true/y_true.max()
        

        delta_ndcg = torch.abs(self.ndcg_at_k(y_true, y_pred, self.k))

        y_pred = y_pred
        pred_diff = y_pred.unsqueeze(1) - y_pred.unsqueeze(0)
        true_diff = true_labels.unsqueeze(1) - true_labels.unsqueeze(0)
            
        S_ij = (true_diff >0).float()
        # RankNet-based loss
        rank_loss = -S_ij * pred_diff
        total_loss = torch.log(1 + torch.exp(rank_loss)) * delta_ndcg
        return total_loss.mean()

    def forward(self, y_pred, true_labels,k = 10):
        self.k = k
        """
        Compute the LambdaRank loss.
        
        y_true: batch of true relevance scores
        y_pred: batch of predicted scores
        """


        return self.forward_(y_pred,true_labels)+self.forward_(y_pred*-1,true_labels*-1)
    

if __name__ == '__main__':
    x= torch.Tensor([0.2,0.81,0.82,0.4]).cuda()
    y = torch.zeros([4,2]).cuda()
    y[:,1] = 0
    y[2,1] = 0
    y[1,1] = 0
    y[1,0] = 0
    y[2,0] = 0
    y[0,1] = 0
    y[3,1] = 0
    loss = LambdaRankLoss(10).cuda()
    print(loss(x,y,sign=-1))