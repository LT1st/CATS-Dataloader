# CATS-Dataloader
preprocess of [CATS](https://vims.cis.udel.edu/cats/setup.html) dataset 
# 发现的问题

数据对没对上去，应该用thermal 和 thermal的深度图

# 加载器

这是一份关于 `CATS2_Dataset` 类的详细文档，该类用于处理和管理一个名为 CATS2 的数据集。以下是对代码的详细解释和文档化：

### `CATS2_Dataset` 类

### 构造函数 `__init__`

- **参数**:
    - `path_2_base` (str): 数据集的绝对路径。
    - `light_con` (str): 光照条件，有三种可选。
    - `pre_process` (bool): 指示是否需要预处理数据，首次加载数据集时需要。
    - `depth_formate` (str): 深度信息的格式。
    - `type_depth` (str): 深度数据的类型。
- **功能**:
    - 初始化数据集的路径、光照条件、深度格式和类型。
    - 根据是否需要预处理，准备所有数据对，并进行预处理。

### 方法 `keep_last_six_elements`

- **功能**:
    - 从给定的路径中提取并保留最后六个元素，生成一个新路径。

### 方法 `split_dict`

- **功能**:
    - 将字典按照指定的比例分割成训练集和验证集。

### 方法 `save_train_val_in_txt`

- **功能**:
    - 将训练集和验证集分别保存到文本文件中。

### 方法 `save_data_root_2_txt`

- **功能**:
    - 将数据对保存到文本文件中，支持不同的保存类型。

### 方法 `pre_process`

- **功能**:
    - 根据深度格式，将数据集中的深度信息从文本文件转换为 PNG 或 NPY 格式。

### 方法 `transfer_depth_to_png`

- **功能**:
    - 将单个深度图像文件从文本格式转换为 PNG 格式。

### 方法 `transfer_dataset_depth_txt_to_png`

- **功能**:
    - 将整个数据集中的深度信息从文本格式转换为 PNG 格式。

### 方法 `transfer_dataset_depth_txt_to_npy`

- **功能**:
    - 将整个数据集中的深度信息从文本格式转换为 NPY 格式。

### 方法 `transfer_depth_file`

- **功能**:
    - 转换单个深度文件的格式。

### 方法 `check_data_scene_ok`

- **功能**:
    - 检查给定文件夹路径下的数据场景是否符合要求。

### 方法 `get_data_root_path`

- **功能**:
    - 获取数据集中所有有效的数据场景路径。

### 方法 `check_and_remove_invalid_entries`

- **功能**:
    - 检查字典中的路径是否有效，并移除无效条目。

### 方法 `get_data_path`

- **功能**:
    - 从给定路径加载所有场景的数据。

### 方法 `prepare_all_data_pairs`

- **功能**:
    - 准备所有数据对，用于后续的数据处理。

### 方法 `get_all_data_pairs`

- **功能**:
    - 获取所有经过验证的数据对。

### 方法 `get_current_scenes`

- **功能**:
    - 获取当前所有经过验证的场景。

### 方法 `check_dataset_ok`

- **功能**:
    - 检查整个数据集是否符合要求。

### 使用示例

- 示例代码展示了如何使用 `CATS2_Dataset` 类来管理和处理数据集。包括初始化类，获取数据对，转换深度信息格式，检查数据集完整性，以及保存数据。

此文档旨在为使用和理解 `CATS2_Dataset` 类提供详细的指导和背景信息。
