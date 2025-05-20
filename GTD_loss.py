import torch
import torch.nn as nn
import torch.nn.functional as F

class LSP_Loss(nn.Module):
    def __init__(self, kernel_type='euclidean', threshold=0.3, epsilon=1e-8, tau=0.1):
        super(LSP_Loss, self).__init__()
        self.kernel_type = kernel_type
        self.threshold = threshold
        self.epsilon = epsilon
        self.tau = tau

    def kernel(self, fi, fj):
        if self.kernel_type == 'euclidean':
            return torch.norm(fi - fj, p=2, dim=-1)
        elif self.kernel_type == 'linear':
            return torch.sum(fi * fj, dim=-1)
        elif self.kernel_type == 'polynomial':
            c = 1  # constant
            d = 3  # degree
            return (torch.sum(fi * fj, dim=-1) + c) ** d
        elif self.kernel_type == 'rbf':
            gamma = 0.5  # gamma parameter
            return torch.exp(-gamma * torch.norm(fi - fj, p=2, dim=-1) ** 2)
        else:
            raise ValueError("Unsupported kernel type")

    def forward(self, student_features, student_adj_matrix, teacher_features, teacher_adj_matrix, removed_nodes=None):
        B, student_node_num, D = student_features.shape
        teacher_node_num = teacher_features.shape[1]

        if student_node_num != teacher_node_num:
            assert student_node_num < teacher_node_num, "Student graph should have fewer nodes compared to teacher graph."
            removed_nodes = torch.tensor(removed_nodes, device=teacher_adj_matrix.device)

        # Preprocess adjacency matrices
        student_adj_matrix = torch.abs(student_adj_matrix)
        teacher_adj_matrix = torch.abs(teacher_adj_matrix)

        student_adj_matrix = (student_adj_matrix - student_adj_matrix.min()) / (student_adj_matrix.max() - student_adj_matrix.min())
        teacher_adj_matrix = (teacher_adj_matrix - teacher_adj_matrix.min()) / (teacher_adj_matrix.max() - teacher_adj_matrix.min())

        student_adj_matrix_binary = (student_adj_matrix > self.threshold).float()
        teacher_adj_matrix_binary = (teacher_adj_matrix > self.threshold).float()

        # Compute kernel matrices
        student_kernel_matrix = self.compute_kernel_matrix(student_features)
        teacher_kernel_matrix = self.compute_kernel_matrix(teacher_features[:, :student_node_num, :])

        # Normalize features for cosine similarity
        student_features_norm = F.normalize(student_features, p=2, dim=-1)

        # Compute cosine similarity and apply temperature scaling
        sim_matrix = torch.matmul(student_features_norm, student_features_norm.transpose(2, 1)) / self.tau

        # Apply log-sum-exp trick for numerical stability
        max_sim = torch.max(sim_matrix, dim=-1, keepdim=True)[0]
        exp_sim_matrix = torch.exp(sim_matrix - max_sim)

        if student_node_num != teacher_node_num:
            teacher_adj_subset = teacher_adj_matrix_binary[:student_node_num, :student_node_num]

            connected_in_teacher = teacher_adj_subset > 0
            indirect_connection = torch.any(
                (teacher_adj_matrix_binary[removed_nodes, :student_node_num].unsqueeze(1) > 0) &
                (teacher_adj_matrix_binary[:student_node_num, removed_nodes].unsqueeze(0) > 0), dim=-1
            )

            connected_or_indirect = connected_in_teacher | indirect_connection

            lsp_loss_numerator = (
                F.kl_div(
                    F.log_softmax(student_kernel_matrix, dim=-1),
                    F.softmax(teacher_kernel_matrix, dim=-1),
                    reduction='none'
                )
                .masked_select(connected_or_indirect)
                .sum()
                * exp_sim_matrix.masked_select(connected_or_indirect)
                .sum()
            )

            count_numerator = connected_or_indirect.sum().item()

            student_connected_not_teacher = (student_adj_matrix_binary > 0) & ~connected_or_indirect

            lsp_loss_denominator = (
                F.kl_div(
                    F.log_softmax(student_kernel_matrix, dim=-1),
                    F.softmax(teacher_kernel_matrix, dim=-1),
                    reduction='none'
                )
                .masked_select(student_connected_not_teacher)
                .sum()
                * exp_sim_matrix.masked_select(student_connected_not_teacher)
                .sum()
            )

            count_denominator = student_connected_not_teacher.sum().item()
        else:
            connected_in_teacher = teacher_adj_matrix_binary > 0

            lsp_loss_numerator = (
                F.kl_div(
                    F.log_softmax(student_kernel_matrix, dim=-1),
                    F.softmax(teacher_kernel_matrix, dim=-1),
                    reduction='none'
                )
                .masked_select(connected_in_teacher)
                .sum()
                * exp_sim_matrix.masked_select(connected_in_teacher)
                .sum()
            )

            count_numerator = connected_in_teacher.sum().item()

            student_connected_not_teacher = (student_adj_matrix_binary > 0) & (teacher_adj_matrix_binary == 0)

            lsp_loss_denominator = (
                F.kl_div(
                    F.log_softmax(student_kernel_matrix, dim=-1),
                    F.softmax(teacher_kernel_matrix, dim=-1),
                    reduction='none'
                )
                .masked_select(student_connected_not_teacher)
                .sum()
                * exp_sim_matrix.masked_select(student_connected_not_teacher)
                .sum()
            )

            count_denominator = student_connected_not_teacher.sum().item()

        numerator_loss = lsp_loss_numerator / count_numerator if count_numerator > 0 else torch.tensor(0.0, device=student_features.device)
        denominator_loss = lsp_loss_denominator / count_denominator if count_denominator > 0 else torch.tensor(1.0, device=student_features.device)

        return numerator_loss / (denominator_loss + self.epsilon)

    def compute_kernel_matrix(self, features):
        B, N, D = features.shape
        feature_diff = features.unsqueeze(2) - features.unsqueeze(1)
        if self.kernel_type == 'euclidean':
            kernel_matrix = torch.norm(feature_diff, p=2, dim=-1)
        elif self.kernel_type == 'linear':
            kernel_matrix = torch.sum(features.unsqueeze(2) * features.unsqueeze(1), dim=-1)
        elif self.kernel_type == 'polynomial':
            c = 1
            d = 3
            kernel_matrix = (torch.sum(features.unsqueeze(2) * features.unsqueeze(1), dim=-1) + c) ** d
        elif self.kernel_type == 'rbf':
            gamma = 0.5
            kernel_matrix = torch.exp(-gamma * torch.norm(feature_diff, p=2, dim=-1) ** 2)
        else:
            raise ValueError("Unsupported kernel type")
        return kernel_matrix



if __name__ == '__main__':

    # Example usage:
    batch_size = 32
    student_node_num = 24
    teacher_node_num = 54
    num_features = 128

    student_features = torch.rand(batch_size, student_node_num, num_features)
    teacher_features = torch.rand(batch_size, teacher_node_num, num_features)
    student_adj_matrix = torch.rand(student_node_num, student_node_num) * 0.2 - 0.1  # Values between -0.1 and 0.1
    teacher_adj_matrix = torch.rand(teacher_node_num, teacher_node_num) * 0.2 - 0.1  # Values between -0.1 and 0.1
    removed_nodes = [0, 1, 17, 18, 21, 22, 23, 24, 27, 28, 4, 5, 8, 9, 12, 13, 38, 39, 42, 43, 44, 45, 33, 34]

    criterion = LSP_Loss(kernel_type='euclidean', threshold=0.5)
    loss = criterion(student_features, student_adj_matrix, teacher_features, teacher_adj_matrix, removed_nodes)
    print(loss)
