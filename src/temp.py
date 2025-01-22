from dataloaders.imagenet_r_subset_spliter import ImagenetR_Spliter
from dataloaders.cifar100_subset_spliter import Cifar100_Spliter

# spliter = ImagenetR_Spliter(client_num=3, task_num=10, private_class_num=10, input_size=224)
spliter = Cifar100_Spliter(client_num=3, task_num=10, private_class_num=10, input_size=32)

spliter.random_split()