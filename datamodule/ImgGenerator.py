from torch.utils.data import Dataset
import torch


class ImgGenerator(Dataset):
    def __init__(self, generate_num=50000, generate_size=(3, 64, 64)):
        """
        初期化処理。
        :param sample_num: 生成するサンプル数（デフォルトは50K）
        :param input_size: 生成画像のサイズ
        """
        self.generate_num = generate_num
        self.generate_size = generate_size

    def __len__(self):
        return self.generate_num

    def __getitem__(self, idx):
        generate_item = torch.zeros(self.generate_size)
        return generate_item


# # test code
# if __name__ == '__main__':
#     sampler = ImgGenerator()
#     print(sampler.__getitem__(0).shape)
