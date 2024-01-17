import os
import numpy as np
import cv2
from tqdm import tqdm
import random
import matplotlib.pyplot as plt



class CATS2_Dataset:
    def __init__(self, path_2_base, light_con='normal',pre_process=False,
                 depth_formate='png',type_depth='png'):
        """

        :param path_2_base: 数据集地址 绝对路径
        :param light_con: 光照情况 三种
        :param pre_process: 是否需要预处理数据，首次加载数据集需要
        :param depth_formate: 深度信息类型
        """
        self.path = path_2_base
        self.light_condition=light_con
        self.prepocess = pre_process
        self.depth_formate = depth_formate
        self.depth_type = type_depth

        if pre_process:
            self.path_dict = self.prepare_all_data_pairs(type="txt")
            self.pre_process()
            self.path_dict = self.prepare_all_data_pairs()
        else:
            self.path_dict = self.prepare_all_data_pairs()

        self.train_dict = None
        self.val_dict = None

    def keep_last_six_elements(self, path):
        # 使用路径分隔符将路径拆分为元素列表
        elements = path.split("\\")  # 如果在 macOS 或 Linux 上运行，请将 "\\" 替换为 "/"

        # 只保留后面的六个元素
        last_six_elements = elements[-6:]

        # 重新组合为新路径
        new_path = "\\".join(last_six_elements)  # 如果在 macOS 或 Linux 上运行，请将 "\\" 替换为 "/"

        return new_path

    def split_dict(self, all_dic=None, train_ratio=0.8):
        """
        将字典按照指定比例拆分为训练集和验证集

        参数:
            - dictionary: 要拆分的原始字典
            - train_ratio: 训练集的比例（0到1之间的浮点数）

        返回值:
            - train_dict: 训练集字典
            - val_dict: 验证集字典
        """
        # 拷贝原始字典，以避免修改原始数据
        original_dict = all_dic.copy()
        # 拆分后的字典
        train_dict = {}
        val_dict = {}
        # 获取原始字典的键列表
        keys = list(original_dict.keys())
        # 随机打乱键的顺序
        random.shuffle(keys)
        # 计算拆分的索引位置
        train_size = int(len(keys) * train_ratio)
        val_size = len(keys) - train_size
        # 按比例拆分字典
        train_keys = keys[:train_size]
        val_keys = keys[train_size:]
        # 填充训练字典
        for key in train_keys:
            train_dict[key] = original_dict[key]
        # 填充验证字典
        for key in val_keys:
            val_dict[key] = original_dict[key]

        return train_dict, val_dict

    def save_train_val_in_txt(self,train_ratio=0.8):
        self.train_dict, self.val_dict = self.split_dict(self.path_dict)
        self.save_data_root_2_txt(type='train_val')

    def save_data_root_2_txt(self, save_path='./', type='all_pic'):

        def save_data_dict_to_file(data_pairs, filename):
            # 排序字典的键，并获取排序后的键列表
            sorted_keys = sorted(data_pairs.keys())
            # 保存排好序的元素到文本文件
            with open(filename, 'w') as file:
                for key in sorted_keys:
                    scene_data = data_pairs[key]
                    n_scene = []
                    for apath in scene_data:
                        n_scene.append( self.keep_last_six_elements(apath))
                    line = ' '.join(n_scene) + '\n'
                    file.write(line)
            print(filename,"列表已成功保存到文本文件。")

        def save_list_to_txt(lst, file_path):
            try:
                with open(file_path, 'w') as file:
                    for item in lst:
                        file.write(str(item) + '\n')
                print("列表已成功保存到文本文件。")
            except IOError:
                print("保存列表到文本文件时出现IO错误。")

        if type == 'train_val':
            save_data_dict_to_file(self.train_dict, 'catsv2_train.txt')
            save_data_dict_to_file(self.val_dict, 'catsv2_val.txt')

        if type == 'all_pic':
            dict = self.path_dict
            save_data_dict_to_file(dict, 'catsv2_all_pic.txt')

        elif type == 'scene_path':
            pdict = self.path_dict.copy()
            roots = []
            for k, v in pdict.items():
                path_this = pdict[k][0]
                path_this, _ = os.path.split(path_this)
                path_this, _ = os.path.split(path_this)
                path_this, _ = os.path.split(path_this)
                roots.append(path_this)

            if roots is None:
                print(" no path available in path")
            save_list_to_txt(roots, 'catsv2_all.txt')

        elif type == 'rgb_pic':
            data_pairs = self.path_dict.copy()
            sorted_keys = sorted(data_pairs.keys())
            # 保存排好序的元素到文本文件
            with open('catsv2_l_rgb.txt', 'w') as file:
                for key in sorted_keys:
                    scene_data = data_pairs[key]
                    line = ' '.join(scene_data) + '\n'
                    file.write(line)


    def save_scene_root_2_txt(self, save_path='./',type='sorted'):
        if type=='sorted':
            pdict = self.path_dict.copy()
            roots = []
            for k,v in pdict.items():
                path_this = pdict[k][0]
                path_this, _ = os.path.split(path_this)
                path_this, _ = os.path.split(path_this)
                roots.append(path_this)

        elif type=='all':
            roots = self.get_data_root_path()

        def save_list_to_txt(lst, file_path):
            try:
                with open(file_path, 'w') as file:
                    for item in lst:
                        file.write(str(item) + '\n')
                print("列表已成功保存到文本文件。")
            except IOError:
                print("保存列表到文本文件时出现IO错误。")

        if roots is None:
            print(" no path available in path")
        save_list_to_txt(roots, 'catsv2_l_rgb.txt')

    def pre_process(self):
        if self.depth_formate=='png':
            self.transfer_dataset_depth_txt_to_png()
        elif self.depth_formate=='npy':
            self.transfer_dataset_depth_txt_to_npy()

    def transfer_depth_to_png_lagacy(self, example_path, tar_file_name, debug=False):
        """
        此版本的归一化方式有问题，导致nan值出现问题
        单个图像转换成png
        :param example_path:
        :return:
        """

        data_path = example_path

        with open(data_path, "r", encoding='latin-1') as file:
            lines = file.readlines()

        # 解析数据并将NaN值设为0
        data = []
        for line in lines:
            line = line.rstrip("\n")  # 去掉行末换行符
            row = line.split(',')
            row = [float(val) if val != "NaN" else 0 for val in row]
            data.append(row)

        # 转换为NumPy数组
        data = np.array(data)

        # 缩放深度数据到合理范围（例如0-255）
        data_min = np.min(data)
        data_max = np.max(data)
        scaled_data = (data - data_min) * (255 / (data_max - data_min))
        scaled_data = scaled_data.astype(np.float32)
        scaled_data = cv2.cvtColor(scaled_data, cv2.COLOR_GRAY2RGB)
        # 转换数据类型为整数

        # name_to_save = os.path.basename(example_path).split(".")[0] +"_"+ "img.png"
        name_to_save = tar_file_name
        # print(name_to_save)

        cv2.imwrite(name_to_save, scaled_data)

        if debug:
            import matplotlib.pyplot as plt
            plt.imshow(data, cmap='viridis')
            plt.colorbar()
            plt.show()

    def transfer_depth_to_png(self, example_path, tar_file_name, debug=False):
        """
        单个图像转换成png
        :param example_path: 源文件路径
        :param tar_file_name: 目标文件名
        :param debug: 是否开启调试模式，以可视化方式显示图像
        """
        data_path = example_path

        with open(data_path, "r", encoding='latin-1') as file:
            lines = file.readlines()

        # 解析数据，使用np.nan表示NaN值
        data = []
        for line in lines:
            line = line.rstrip("\n")  # 去掉行末换行符
            row = line.split(',')
            row = [float(val) if val != "NaN" else np.nan for val in row]
            data.append(row)

        # 转换为NumPy数组
        data = np.array(data)

        # 创建掩码标记NaN值的位置
        mask = np.isnan(data)

        # 将NaN暂时替换为0
        data[mask] = 0

        # 缩放深度数据到合理范围（例如0-255），忽略NaN位置
        data_min = np.nanmin(data)
        data_max = np.nanmax(data)
        scaled_data = (data - data_min) / (data_max - data_min) * 255
        scaled_data = scaled_data.astype(np.uint8)

        # 将掩码位置的值重新设为0
        scaled_data[mask] = 0

        # 如果需要，将灰度图转为RGB
        scaled_data = cv2.cvtColor(scaled_data, cv2.COLOR_GRAY2RGB)

        # 保存图像
        cv2.imwrite(tar_file_name, scaled_data)

        # 如果开启了调试模式，显示图像
        if debug:
            plt.imshow(scaled_data, cmap='gray')
            plt.colorbar()
            plt.show()

    # 示例调用
    # transfer_depth_to_png("example_path.txt", "output.png", debug=True)

    # %%
    def transfer_dataset_depth_txt_to_png(self):
        """
        把数据集的所有txt文件转换,同时更改字典
        :param all_data:
        :return:
        """
        self.depth_type = 'png'

        all_data = self.path_dict       # 获取当前路径字典
        data_paris = all_data.copy()    # 复制一份路径字典，以避免直接修改原始字典
        for data_pair in tqdm(data_paris):
            # print(data_pair)
            example_paths = data_paris[data_pair]       # 获取当前数据对的路径
            example_path = example_paths[1]             # 选取深度图的路径
            base_file = os.path.dirname(example_path)   # 获取文件的基础路径和文件名
            base_path = os.path.basename(example_path)
            if 'png' in base_path : # 根据文件类型确定需要转换的文件路径
                example_path = os.path.join(os.path.dirname(example_path) ,'gt_disparity.txt')
            elif 'npy' in base_path:
                example_path = os.path.join(os.path.dirname(example_path) ,'gt_disparity.txt')

            # 构造目标文件名
            tar_file_name = os.path.basename(example_path).split(".")[0] + "_" + "norm" + "_" + "img.png"
            # tar_file_name = os.path.join(base_file, tar_file_name)
            # 创建新的路径列表，包含转换后的文件路径
            temp_path = example_paths[0:2]
            transfered_example_paths = os.path.join(base_file, tar_file_name)
            temp_path.append(transfered_example_paths)
            # print(transfered_example_paths,temp_path,tar_file_name)
            data_paris[data_pair] = temp_path   # 更新字典中的路径
            # print(example_path, tar_file_name)
            self.transfer_depth_to_png(example_path, transfered_example_paths)      # 调用函数将txt文件转换为png格式

        return data_paris

    def transfer_dataset_depth_txt_to_npy(self):
        """
        把数据集的所有txt文件转换
        :param all_data:
        :return:
        """
        self.depth_type = 'npy'
        all_data = self.path_dict
        data_paris = all_data.copy()
        for data_pair in tqdm(data_paris):
            # print(data_pair)
            example_paths = data_paris[data_pair]
            example_path = example_paths[2]
            base_file = os.path.dirname(example_path)
            tar_file_name = os.path.basename(example_path).split(".")[0] + "norm" + ".npy"

            temp_path = example_paths[0:2]
            transfered_example_paths = os.path.join(base_file, tar_file_name)
            temp_path.append(transfered_example_paths)
            # print(transfered_example_paths,temp_path,tar_file_name)
            data_paris[data_pair] = temp_path

            transfered_example_paths = temp_path.append(tar_file_name)
            data_paris[data_pair] = transfered_example_paths

            self.transfer_depth_file(example_path, tar_file_name)

        return data_paris

    def transfer_depth_file(self, example_path, target_path):
        """
        转换 一个 文件格式，保存到原始i位置
        :param example_path:
        :return: None
        """
        # print(example_path)
        data_path = example_path

        with open(data_path, "r") as file:
            lines = file.readlines()

        # 解析数据并将NaN值设为0
        data = []
        for line in lines:
            line = line.rstrip("\n")  # 去掉行末换行符
            row = line.split(',')
            row = [float(val) if val != "NaN" else 0 for val in row]
            data.append(row)

        # 转换为NumPy数组
        data = np.array(data)

        # 缩放深度数据到合理范围（例如0-255）
        data_min = np.min(data)
        data_max = np.max(data)
        scaled_data = (data - data_min) * (255 / (data_max - data_min))

        # 转换数据类型为整数
        scaled_data = scaled_data.astype(np.uint16)

        np.save(target_path, scaled_data)
    def check_data_scene_ok(self, path_folder):
        return True

    def get_data_root_path(self):
        data_root = []  # 包含左右红外、深度
        path_to_datafile = self.path

        item_sen = "Indoor"
        entity_path_all = os.path.join(path_to_datafile, item_sen)
        if os.path.isdir(entity_path_all):
            for entity_item in os.listdir(entity_path_all):
                entity_path = os.path.join(entity_path_all, entity_item)
                if os.path.isdir(entity_path):
                    for scene_item in os.listdir(entity_path):
                        scene_path = os.path.join(entity_path, scene_item)
                        if os.path.isdir(scene_path) and self.check_data_scene_ok(scene_path):
                            data_root.append(scene_path)
        item_sen = "Outdoor"
        entity_path_all = os.path.join(path_to_datafile, item_sen)
        if os.path.isdir(entity_path_all):
            for entity_item in os.listdir(entity_path_all):
                entity_path = os.path.join(entity_path_all, entity_item)
                if os.path.isdir(entity_path):
                    for scene_item in os.listdir(entity_path):
                        scene_path = os.path.join(entity_path, scene_item)
                        if os.path.isdir(scene_path)  and self.check_data_scene_ok(scene_path):
                            data_root.append(scene_path)
        # data_root = self.check_and_remove_invalid_entries(data_root)
        return data_root

    def check_and_remove_invalid_entries(self, data_dict):
        """
        检查字典中地址文件是否存在并删除不存在的条目

        Args:
            data_dict (dict): 包含地址文件路径的字典

        Returns:
            dict: 更新后的字典，不包含不存在的地址文件条目
        """
        valid_entries = {}
        for key, value in data_dict.items():
            all_paths_valid = True
            for path in value:
                # 检查地址文件是否存在
                if not os.path.exists(path):
                    print(f"地址文件不存在: {path}，已从字典中删除")
                    all_paths_valid = False
                    break

            if all_paths_valid:
                valid_entries[key] = value

        return valid_entries

    def get_data_path(self,path_to_datafile, item_sen="Indoor", depth_type='png'):
        """
        从路径中加载所有场景的数据
        :param path_to_datafile: 指向路径
        :param item_sen: 室内还是室外
        :param depth_type: 深度数据类型
        :return:
        """
        depth_type = self.depth_type
        data_paris = {}  # 包含左右红外、深度

        entity_path_all = os.path.join(path_to_datafile, item_sen)
        if os.path.isdir(entity_path_all):
            for entity_item in os.listdir(entity_path_all):
                entity_path = os.path.join(entity_path_all, entity_item)
                if os.path.isdir(entity_path):
                    for scene_item in os.listdir(entity_path):
                        scene_path = os.path.join(entity_path, scene_item )
                        if os.path.isdir(scene_path):
                            # scene_name = os.path.basename(entity_item)+'_'+os.path.basename(scene_path)
                            scene_name = entity_item+"_"+scene_item
                            # todo 这里的三种图用哪个？
                            scene_left_thermal = os.path.join(scene_path, "rectified", 'thermal',"left_thermal_default.png")
                            # scene_right_thermal = os.path.join(scene_path, "rectified",'thermal', "right_thermal_default.png")
                            if depth_type=='png':
                                # gt_disparity_norm_img.png
                                scene_depth = os.path.join(scene_path, "gt_disparity",'thermal',"gt_disparity_norm_img.png")
                            elif depth_type=='txt':
                                scene_depth = os.path.join(scene_path, "gt_disparity",'thermal',"gt_disparity.txt")
                            else:
                                scene_depth = os.path.join(scene_path, "gt_disparity",'thermal',"gt_disparity_norm.npy")
                            # print(scene_name)
                            data_paris[scene_name]=[scene_left_thermal, scene_depth]
        return data_paris

    def prepare_all_data_pairs(self, type=None):
        if type is not None:
            data_paris_Indoor = self.get_data_path(self.path, 'Indoor', type)
            data_paris_Outdoor = self.get_data_path(self.path, 'Outdoor', type)

            data_paris_Indoor.update(data_paris_Outdoor)
        else:
            data_paris_Indoor = self.get_data_path(self.path, 'Indoor', self.depth_formate)
            data_paris_Outdoor = self.get_data_path(self.path, 'Outdoor', self.depth_formate)

            data_paris_Indoor.update(data_paris_Outdoor)
        # data_paris_Indoor = self.check_and_remove_invalid_entries(data_paris_Indoor)
        print(data_paris_Outdoor)
        return data_paris_Indoor

    def get_all_data_pairs(self):
        self.path_dict = self.check_and_remove_invalid_entries(self.path_dict)
        return self.path_dict

    def get_current_scenes(self):
        self.path_dict = self.check_and_remove_invalid_entries(self.path_dict)
        return self.path_dict.values()

    def check_dataset_ok(self):
        self.path_dict = self.check_and_remove_invalid_entries(self.path_dict)




if __name__ == "__main__":

    # -------------- 需要重新生成depth时 --------------------
    path=os.getcwd()  # 获取当前目录
    print(path)
    cats2 = CATS2_Dataset(path, pre_process=True, depth_formate='png', type_depth='png')


    # -------------- 只需要修改目录时 --------------------
    # path=os.getcwd()  # 获取当前目录
    # print(path)
    # cats2 = CATS2_Dataset(path)
    # # data_paris_Indoor = cats2.get_data_path('Indoor','png')
    # # data_paris_Outdoor = cats2.get_data_path('Outdoor','png')
    # #
    # # data_paris_Indoor.update(data_paris_Outdoor)
    # # print(data_paris_Indoor)
    # print(cats2.get_all_data_pairs())
    # # cats2.transfer_dataset_depth_txt_to_png()
    # cats2.check_dataset_ok()
    # cats2.save_scene_root_2_txt()
    # cats2.save_data_root_2_txt(type='scene_path')
    # cats2.save_data_root_2_txt()
    #
    # cats2.save_train_val_in_txt()
