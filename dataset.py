import torch

class MoonBoardWeightedDataset(torch.utils.data.Dataset):
    def __init__(self, board_tensor, is_bench_tensor, benchmark_weight=10):
        self.board_tensor = board_tensor
        # TODO: check weight is a scalar probably
        assert board_tensor.size(0) == is_bench_tensor.size(0), f"First dimension of tensors do not match! {board_tensor.shape[0]} != {is_bench_tensor.shape[0]}"
        assert is_bench_tensor.dim() == 1

        self.is_bench_tensor = torch.where(is_bench_tensor > 0, benchmark_weight, 1).reshape(is_bench_tensor.size(0), 1)

    def __len__(self):
        return self.board_tensor.size(0) 

    def __getitem__(self, idx):
        board = self.board_tensor[idx]
        weight = self.is_bench_tensor[idx]
        return board, weight