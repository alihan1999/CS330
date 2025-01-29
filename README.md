# CS330 2023

Homework Solutions for The Multitask/Meta Learning course offered by Stanford
# HOMEWORK 2 ✍️
## OBJECTIVES 🎯
- __Studying [Protonet](https://arxiv.org/abs/1703.05175)__
- __Introduction to Optimization Based Meta Learning and [MAML](https://arxiv.org/abs/1703.03400)__
- __Comparing Protonet and MAML under different settings at Meta-Test time__
- __Running Experiments with MAML under different settings including the magnitude of the inner learning rate, the number of inner iterations and learning inner rates__
## Coding Problems
- **Implementing the _step method of Protonet**
  python
  def _step(self, task_batch):
        loss_batch = []
        accuracy_support_batch = []
        accuracy_query_batch = []
        for i, task in enumerate(task_batch):
  
            images_support, labels_support, images_query, labels_query = task
           
            images_support = images_support.to(self.device)
            labels_support = labels_support.to(self.device)
            images_query = images_query.to(self.device)
            labels_query = labels_query.to(self.device)
            
            support_features = self._network(images_support)
              
            prototypes = []

            for label in labels_support.unique():
                prototypes.append(support_features[labels_support==label].mean(0))       
            prototypes = torch.stack(prototypes)
            
            query_features = self._network(images_query)
           
            query_dist = -torch.cdist(query_features,prototypes)
            support_dist = -torch.cdist(support_features,prototypes)

            loss_batch.append(F.cross_entropy(query_dist,labels_query))
            accuracy_support_batch.append(util.score(support_dist,labels_support))
            accuracy_query_batch.append(util.score(query_dist,labels_query))
            
        return (
            torch.mean(torch.stack(loss_batch)),
            np.mean(accuracy_support_batch),
            np.mean(accuracy_query_batch)
        ) ```
