import os
from collections import OrderedDict
import torch
from PIL import Image
from torch import Tensor
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import logging
from pathlib import Path
from scipy.spatial.transform import Rotation
from skimage import io
from skimage.transform import resize

from torchvision import transforms
import cv2

# from tools.rotation_conversions import ortho6d_from_rotation_matrix, rotation_matrix_from_ortho6d, tmat_from_9d

# from dupla_renderers.pytorch3d.utilities import kneefit_to_pytorch3d_renderer
from dupla_renderers.pytorch3d import AnatomyCT, AnatomySTL, Camera, CTRenderer, STLRenderer, Scene
from pytorch3d.transforms import rotation_6d_to_matrix, matrix_to_rotation_6d
from dl_models_leomed.tools.rotation_conversions import pose_to_tmat


class SiplaDataset18(Dataset):
    """
    returns xrayImage, pose, addInfo
    in the knee data, the pose is given through 3x4 rotation matrices, we transform to 18pose
    order given by `self.poseCols`
    """
    DTYPE_TORCH = torch.float32
    DTYPE_NP = np.float32
    R_TOL = 1e-04  # how close need 2 values be to be considered equal by np.all_close()
    A_TOL = 1e-02  # 0.1 mm or 0.1 degree

    def __init__(
            self,
            anatomy_type,
            outSizePxl=512,
            standardizeCalib: bool = True,
            imageType: str = "Both",
            expectedSizePxl: int = 1000,  # the original image pixel size
            renderer_type='CT',
            add_noise=False,
            with_synth=False,
            transform=None,
            data_length=None,
            normalize_translation=False,
            mask_femur=True,
            mask_tibia=True,
            mask_binary=True,
            femur_name=None,
            tibia_name=None,
            only_test=False,

    ):
        """
        Args:
            df: the input dataframe
            transformer: the TransformerPipeline to apply
                ..warning:: no fitting is done currently
            outSizePxl: resize xray to this size, or direct output resolution if we are rendering
            imageType: "xray" for original image, "synthBoth", "synthFemur", "synthTibia" for
                renderings
            standardizeCalib: change calib to principal point of 0,0 by changing the translation
                components of the pose. That is an approximation, but should be decent for our
                application.
        """
        super().__init__()
        self.anatomy_type = anatomy_type
        self.add_noise = add_noise
        self.with_synth = with_synth
        self.normalize_translation=normalize_translation
        assert self.anatomy_type in (
            "Bone", "Implant", "bone", "implant"), "anatomy_type must be 'Bone', 'Implant', 'bone' or 'implant'"
        self.anatomy_type = self.anatomy_type.lower()  # make sure anatomy_type is in lower case
        self.femur_name=femur_name
        self.tibia_name=tibia_name
        if self.anatomy_type == 'bone':
            self.database_label = 'kneefit'
        else:
            self.database_label = 'flumatch'
        self.outSizePxl = outSizePxl
        self.expectedSizePxl = expectedSizePxl
        self.data_length=data_length
        self.bone_image_folder = r'/cluster/work/ifb_lmb/2d_3d_registration/image_collection_all/SIPLA_Bone/Images'
        self.implant_image_folder = r'/cluster/work/ifb_lmb/2d_3d_registration/image_collection_all/SIPLA_Implant/Images'
        self.df = pd.read_csv(rf'/cluster/home/wangjinh/myproject/dl_models_leomed/sipla_data_combined_jinhao.csv')

        self.sipla_model_folder = rf'/cluster/work/ifb_lmb/2d_3d_registration/image_collection_all/SIPLA_models_all/sipla_models'
        self.premask_folder = rf'/cluster/home/wangjinh/premask(synthetic images)'
        # load self default transform
        self.default_transform = transforms.Compose([
            transforms.Resize((outSizePxl, outSizePxl)),  # Resize to model input size
        ])
        self.only_test = only_test
        self._validate_df(self.df)
        print(f"Dataframe loaded and validated. Shape: {self.shape}")
        # filter data, only keep data that fits anatomy type and anatomy name
        self._filter_dataframe()

        self.outSizePxl = outSizePxl
        self.transform = transform
        self.imageType = imageType
        

        self._determine_device()
        self._check_device()
        # we need to standardize camera setting otherwise a network with them should be used.
        self.standardizeCalib = standardizeCalib
        # standardValues
        self.stdCal = {"focal": 980, "x": 0., "y": 0.}
        self.cal_principal_point_h = 0
        self.cal_principal_point_v = 0
        self.cal_focal_length = 980
        self.cal_pixel_size = 0.28
        
        self.mask_femur=mask_femur
        self.mask_tibia=mask_tibia
        self.mask_binary=mask_binary
        # prepare tools for rendering synthetic images
        cam = Camera(
            "camera_1",
            (0, 0, 0),
            (0, 0, 1),
            (0, 1, 0),
            1000 * self.cal_pixel_size,
            1000 * self.cal_pixel_size,
            self.cal_principal_point_h,
            self.cal_principal_point_v,
            self.cal_focal_length,
        )
        their_world_to_yours = torch.tensor(
            [
                [1, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 1, 980],
                [0, 0, 0, 1],
            ],
            dtype=self.DTYPE_TORCH,
        )[None]
        self.renderer_type = renderer_type
        if self.renderer_type == 'CT':
            renderer = CTRenderer(device="cuda")
        else:
            renderer = STLRenderer(device="cuda")

        self.renderer = renderer
        self.cam = cam
        self.their_world_to_yours = their_world_to_yours

    def _initialize_scene(self):
        scene_sipla = Scene()
        if self.renderer_type == 'CT':
            scene_sipla.add_anatomies(self.femur_ct)
            scene_sipla.add_anatomies(self.tibia_ct)
        else:
            scene_sipla.add_anatomies(self.femur_stl)
            scene_sipla.add_anatomies(self.tibia_stl)
        scene_sipla.add_cameras(self.cam)

        self.renderer.bind_scene(scene_sipla)

    def _determine_device(self):
        if torch.cuda.is_available():
            self.device = 'cuda'
            print("CUDA device is available. Setting device to cuda")
        else:
            raise ValueError("CUDA device is not available, please select a different device.")

    def _check_device(self):
        if self.device == 'cuda':
            print("CUDA device is available. Device is set correctly")
        elif self.device == 'cpu':
            raise ValueError("CPU is not allowed, please use a CUDA device.")
        else:
            raise ValueError(f"Unknown device type: {self.device}. Please select 'cuda'.")

    def _validate_df(self, df):
        """gives required fields in the input dataframe, but also order of output pose"""
        self.poseCols = (
            "femur_tx",
            "femur_ty",
            "femur_tz",
            "femur_rxx",
            "femur_rxy",
            "femur_rxz",
            "femur_ryx",
            "femur_ryy",
            "femur_ryz",
            "tibia_tx",
            "tibia_ty",
            "tibia_tz",
            "tibia_rxx",
            "tibia_rxy",
            "tibia_rxz",
            "tibia_ryx",
            "tibia_ryy",
            "tibia_ryz",
        )
        self.femurCol = "femur_nii"
        self.tibiaCol = "tibia_nii"
        self.addCols = (
            "cal_focal_length",
            "cal_mm_per_pxl",
            "cal_principalp_x",
            "cal_principalp_y",
            self.femurCol,
            self.tibiaCol,
        )
        """check whether all required fields are present"""
        for col in self.poseCols + ("img_path",) + self.addCols:
            assert col in df.columns, "Column {} does not exist in Dataframe".format(col)

    def __len__(self):
        return len(self.available_ids)

    def _filter_dataframe(self):
        """using anatomy_type to filter the data"""
        test_ids = set()
        try:
            df_test_ids = pd.read_csv(rf'/cluster/home/wangjinh/myproject/dl_models_leomed/datasets/test_ids.csv')
            if not df_test_ids.empty:
                filtered = df_test_ids[df_test_ids['anatomy_type'].str.lower() == self.anatomy_type.lower()]
                test_ids = set(filtered['id'].tolist())
                print(f"Loaded {len(test_ids)} test ids for anatomy_type: {self.anatomy_type}")
        except FileNotFoundError:
            print("Warning: test_ids.csv not found. No test ids will be excluded.")

        self.df = self.df[
            (self.df['type'].str.lower() == self.anatomy_type)
        ]

        if not self.df.empty and test_ids:
            if self.only_test:
                self.df = self.df[self.df['id'].isin(test_ids)]
            else:
                self.df = self.df[~self.df['id'].isin(test_ids)]

        self.min_selected_id = 0
        if self.femur_name is not None and self.tibia_name is not None:
            self.df = self.df[
                (self.df['femur_nii'].str.contains(self.femur_name)) &
                (self.df['tibia_nii'].str.contains(self.tibia_name)) &
                (self.df['type'].str.lower() == self.anatomy_type)
                ]
            # \u68c0\u67e5\u662f\u5426\u6709\u7b26\u5408\u6761\u4ef6\u7684\u6570\u636e
            if not self.df.empty:
                # \u83b7\u53d6\u8fc7\u6ee4\u540e\u7684\u6700\u5c0fid\u503c
                self.min_selected_id = self.df['id'].min()
                print(f"The minimum selected ID is: {self.min_selected_id}")
            else:
                self.min_selected_id = None
                print("No matching rows found for the given anatomy and paths.")
        self.available_ids = sorted(self.df['id'].tolist())
        print(rf'df has {len(self.df)} rows that matches the anatomy')
        # breakpoint()
        # randomly choose n samples
        # self.df = self.df.sample(n=1000).reset_index(drop=True)

    @property
    def shape(self):
        return self.df.shape

    def __getitem__(self, idx: int):
        """
        Args:
            idx - either int for one sample or list of ints
        Returns:
          inp: typically the x-ray image transformed (determined by _get_in)
          target: typically the pose (determined by _get_out)
          addInfo: additional information
          ..note:: inp and target are converted to self.DTYPE_NP
        Raises:
          FileNotFoundError when image_path is not found
        """
        selected_id = int(self.available_ids[idx])
        assert self.imageType in ("Both", "Femur", "Tibia", "both", "femur", "tibia")
        ds = self.df[(self.df['id'] == selected_id) & (self.df['type'] == self.anatomy_type)]

        assert len(ds) == 1, f"Expected one row, but got {len(ds)}. selected idx: {selected_id}, min_id:{self.min_selected_id} anatomy_type: {self.anatomy_type}"
        # load pose
        original_target_pose = np.array(list(ds[pose_c] for pose_c in self.poseCols)).T
        calDict = {
            "cal_focal_length": ds["cal_focal_length"].values[0],
            "cal_principalp_x": ds["cal_principalp_x"].values[0],
            "cal_principalp_y": ds["cal_principalp_y"].values[0],
            "cal_mm_per_pxl": ds["cal_mm_per_pxl"].values[0],
        }
        # adjust pose from one calibration to the other
        if self.standardizeCalib:
            standardized_target_pose = self.standardize_pose(original_target_pose, calDict)
            standardized_target_pose = standardized_target_pose.astype(self.DTYPE_NP).squeeze()
            standardized_target_pose = torch.tensor(standardized_target_pose)
        else:
            raise ValueError('you are not using standardized_target_pose, are you sure?')
        # add info should have: original calib settings, original true pose, true mask, true pose with noise synthetic image with/without noise
        addInfo = {
            "cal_focal_length": ds["cal_focal_length"].values[0],
            "cal_principalp_x": ds["cal_principalp_x"].values[0],
            "cal_principalp_y": ds["cal_principalp_y"].values[0],
            "cal_mm_per_pxl": ds["cal_mm_per_pxl"].values[0],
            'standardized_target_pose': standardized_target_pose,
            'original_target_pose': original_target_pose,
            'selected idx': selected_id,
            # 'original_target_pose': [None],
            # 'true_mask': [None],
            # 'synthetic_image': [None],
            # 'standardized_target_pose_noised': [None],
            # 'synthetic_image_noised': [None],
        }
        # load xray image and resize to self.outSizePxlPxl or create rendering
        image_path = ds['img_path'].values[0]

        image_name = image_path.split('/')[-1]
        # converting .tiff to .tif
        if not image_name.endswith('.tif'):
            image_name = os.path.splitext(image_name)[0] + '.tif'

        if self.anatomy_type == "bone":
            img_name = os.path.join(self.bone_image_folder, image_name)
        else:
            # \u5c06 .tif \u6269\u5c55\u540d\u66f4\u6539\u4e3a .tiff
            if image_name.endswith(".tif"):
                image_name = image_name.replace(".tif", ".tiff")
            img_name = os.path.join(self.implant_image_folder, image_name)
        # default transformer takes 8 bit image
        image = Image.open(img_name)
        image = np.array(image).astype(np.float32)  # \u8f6c\u6362\u4e3a\u6d6e\u70b9\u6570\u4ee5\u4fbf\u5f52\u4e00\u5316
        image /= 65535.0  # \u5c0616\u4f4d\u56fe\u50cf\u5f52\u4e00\u5316\u5230[0, 1]
        assert image.shape[0] == self.expectedSizePxl
        assert image.shape[1] == self.expectedSizePxl
        if image is None:
            raise FileNotFoundError(f"Image {img_name} not found")

        # create synthetic image if required
        if self.with_synth:
            synth_image = self.create_rendering(ds=ds, target_pose=standardized_target_pose) if self.with_synth else None
            addInfo['synthetic_image'] = synth_image
        else:
            synth_image=None
        # load and binarize mask image
        femur_mask_path = os.path.join(self.premask_folder, f'{self.database_label}_femur_{selected_id}_syn.png')
        tibia_mask_path = os.path.join(self.premask_folder, f'{self.database_label}_tibia_{selected_id}_syn.png')

        femur_mask = Image.open(femur_mask_path)
        tibia_mask = Image.open(tibia_mask_path)

        femur_mask = np.array(femur_mask)
        tibia_mask = np.array(tibia_mask)
        if self.mask_femur==True and self.mask_tibia==True:
            combined_mask = np.clip(femur_mask + tibia_mask, 0, 255)
        elif self.mask_femur==True and self.mask_tibia==False:
            combined_mask = np.clip(femur_mask, 0, 255)
        elif self.mask_femur==False and self.mask_tibia==True:
            combined_mask = np.clip(tibia_mask, 0, 255)
        else:
            raise ValueError('no mask, are you sure?')

        
        if self.mask_binary:
            combined_mask = (combined_mask > 0).astype(np.float32)
        else:
            combined_mask = combined_mask.astype(np.float32) / 255.0  # keep original mask values in [0,1]
        # reverse color to simulate X-ray
        mask_tensor = torch.tensor(1 - combined_mask, dtype=self.DTYPE_TORCH).unsqueeze(0)

        # apply transform to the image and mask
        if self.transform:
            image_tensor = torch.tensor(image, dtype=self.DTYPE_TORCH).unsqueeze(0)
            image = self.transform(image_tensor).type(self.DTYPE_TORCH)
            mask_tensor = self.transform(mask_tensor).type(self.DTYPE_TORCH)

        # create a black layer and concatenate with the target image
        black_layer = torch.zeros_like(image)
        combined_image = torch.cat((image, mask_tensor,image), dim=0)

        # update addInfo
        addInfo['true_mask'] = mask_tensor

        output_pose = standardized_target_pose
        # Normalization based on mean and std
        if self.normalize_translation:
            output_pose[0] = (output_pose[0] - 0) / 40
            output_pose[1] = (output_pose[1] + 0) / 20
            output_pose[2] = (output_pose[2] + 830) / 40
            output_pose[9] = (output_pose[9] + 0) / 40
            output_pose[10] = (output_pose[10] + 50) / 40
            output_pose[11] = (output_pose[11] + 830) / 40

        if self.with_synth:
            if self.add_noise:
                tmat_femur, tmat_tibia = pose_to_tmat(standardized_target_pose)
                noised_tmat_femur, noised_tmat_tibia = self.add_pose_noise(tmat_femur, tmat_tibia)
                noised_pose_vector = self.get_pose_vector(noised_tmat_femur, noised_tmat_tibia)
                addInfo['standardized_target_pose_noised'] = noised_pose_vector
                addInfo['synthetic_image_noised'] = self.create_rendering(ds=ds, target_pose=noised_pose_vector)
                output_pose = noised_pose_vector

        return combined_image, output_pose, addInfo

    def get_pose_vector(self, tmat_femur, tmat_tibia):
        femur_translation = tmat_femur[:, :3, 3].view(-1)
        femur_rotation_matrix = tmat_femur[:, :3, :3]
        femur_rotation_6d = matrix_to_rotation_6d(femur_rotation_matrix)

        tibia_translation = tmat_tibia[:, :3, 3].view(-1)
        tibia_rotation_matrix = tmat_tibia[:, :3, :3]
        tibia_rotation_6d = matrix_to_rotation_6d(tibia_rotation_matrix)

        # Normalization based on mean and std
        if self.normalize_translation:
            femur_translation[0] = (femur_translation[0] - 0) / 40
            femur_translation[1] = (femur_translation[1] + 0) / 20
            femur_translation[2] = (femur_translation[2] + 830) / 40
            tibia_translation[0] = (tibia_translation[0] + 0) / 40
            tibia_translation[1] = (tibia_translation[1] + 50) / 40
            tibia_translation[2] = (tibia_translation[2] + 830) / 40

        pose_vector = torch.cat(
            [femur_translation, femur_rotation_6d.view(-1), tibia_translation, tibia_rotation_6d.view(-1)], dim=0)
        return pose_vector

    def add_pose_noise(self, tmat_femur, tmat_tibia):
        """
        Adds uniform noise each translation (+/- in mm)
        so a value of 50 will add [-50, +50] to the translation components
        "rotationNoise" - and around a random axis with unform angle (in rad)

        Adds uniform noise to the translation components (+/- 30 mm)
        and rotation components (0-10 degrees) around a random axis.
        """
        noised_tmat_femur = tmat_femur.clone()
        noised_tmat_tibia = tmat_tibia.clone()
        translation_noise = np.random.uniform(-30, 30, 3)  # \u5e73\u79fb\u566a\u58f0
        rotation_angle = np.random.uniform(0, np.deg2rad(10))  # \u65cb\u8f6c\u89d2\u5ea6\u566a\u58f0\uff0c\u8f6c\u6362\u4e3a\u5f27\u5ea6
        rotation_axis = np.random.uniform(-1, 1, 3)  # \u968f\u673a\u65cb\u8f6c\u8f74
        rotation_axis /= np.linalg.norm(rotation_axis)  # \u5f52\u4e00\u5316\u65cb\u8f6c\u8f74

        # \u521b\u5efa\u65cb\u8f6c\u77e9\u9635
        rotation_matrix = Rotation.from_rotvec(rotation_angle * rotation_axis).as_matrix()

        # \u6dfb\u52a0\u566a\u58f0\u5230 femur \u53d8\u6362\u77e9\u9635
        noised_tmat_femur[:, :3, 3] += torch.tensor(translation_noise, dtype=self.DTYPE_TORCH)
        noised_tmat_femur[:, :3, :3] = torch.matmul(torch.tensor(rotation_matrix, dtype=self.DTYPE_TORCH),
                                                    noised_tmat_femur[:, :3, :3])

        # \u6dfb\u52a0\u566a\u58f0\u5230 tibia \u53d8\u6362\u77e9\u9635
        noised_tmat_tibia[:, :3, 3] += torch.tensor(translation_noise, dtype=self.DTYPE_TORCH)
        noised_tmat_tibia[:, :3, :3] = torch.matmul(torch.tensor(rotation_matrix, dtype=self.DTYPE_TORCH),
                                                    noised_tmat_tibia[:, :3, :3])

        return noised_tmat_femur, noised_tmat_tibia

    def create_rendering(self, ds, target_pose, addNoise=False, imageType='Both'):
        """load the anatomies from the subset of the dataset and render them in target pose
        after adding noise if applicable
        all inputs must have the same number of elements

        Returns synthetic image and target with noise added
        """
        if self.with_synth:
            self.femur_ct_name = ds['femur_nii'].values[0].split('/')[-1]
            self.tibia_ct_name = ds['tibia_nii'].values[0].split('/')[-1]
            self.femur_stl_name = ds['femur_stl'].values[0].split('\\')[-1]
            self.tibia_stl_name = ds['tibia_stl'].values[0].split('\\')[-1]

            self.femur_ct = AnatomyCT.load_data(os.path.join(self.sipla_model_folder, self.femur_ct_name))
            self.tibia_ct = AnatomyCT.load_data(os.path.join(self.sipla_model_folder, self.tibia_ct_name))
            self.femur_stl = AnatomySTL.load_data(os.path.join(self.sipla_model_folder, self.femur_stl_name))
            self.tibia_stl = AnatomySTL.load_data(os.path.join(self.sipla_model_folder, self.tibia_stl_name))

            # transform target pose to tmat:
            assert target_pose.shape[0] == 18
            tmat_femur, tmat_tibia = pose_to_tmat(target_pose)
            # we are rendering with synthBoth, synthFemur or synthTibia
            self._initialize_scene()
            self.set_model_matrix(self.femur_ct, tmat_femur, self.their_world_to_yours)
            self.set_model_matrix(self.tibia_ct, tmat_tibia, self.their_world_to_yours)

            try:
                foreground_efficient_renderer = self.renderer.render_efficient_memory(
                    0, self.outSizePxl, self.outSizePxl, binary=False
                )[:, :, :].detach().cpu().numpy()
            except torch._C._LinAlgError as e:
                print(f"Matrix inversion failed: {e}")
                print(f"Camera transformation matrix: {self.cam.tmats}")
                breakpoint()

            assert np.max(foreground_efficient_renderer) <= 1.0, rf'{np.max(foreground_efficient_renderer)}'
            assert np.min(foreground_efficient_renderer) >= 0.0, rf'{np.min(foreground_efficient_renderer)}'
            return foreground_efficient_renderer
        else:
            return None

    def set_model_matrix(self, ct_data, tmat, their_world_to_yours):
        ct_data.set_model_matrix(tmat, is_yours=False, theirs_to_yours=their_world_to_yours)

    def standardize_pose(self, pose, calDict):
        """very simple approximation, ignored rotation"""

        frac_pp = np.abs(pose[:, 2]) / calDict["cal_focal_length"]
        # adding effect of pixel size in standardization
        delta_pixel_size = self.cal_pixel_size - calDict["cal_mm_per_pxl"]
        frac_pixel_effect = 1 / (1 + delta_pixel_size / calDict["cal_mm_per_pxl"])

        assert frac_pp > 0
        assert frac_pixel_effect > 0

        pose = pose.copy()


        pose[:, 0] += calDict["cal_principalp_x"] * frac_pp
        pose[:, 1] += calDict["cal_principalp_y"] * frac_pp
        pose[:, 2]  = pose[:, 2] * frac_pixel_effect  # z


        pose[:, 9] += calDict["cal_principalp_x"] * frac_pp
        pose[:, 10] += calDict["cal_principalp_y"] * frac_pp
        pose[:, 11] = pose[:, 11] * frac_pixel_effect  # z

        return pose

    def check_sample(self, sample):
        inp, out, addInfo = sample
        # print(out, out.shape, len(self.poseCols))
        assert out.shape[-1] == len(self.poseCols), f"Got unexpected shape {out.shape}"
        assert inp.max() <= 1, f"Input max value {inp.max()} exceeds 1"
        assert inp.min() >= 0, f"Input min value {inp.min()} is below 0"
        assert inp.shape[-3] == 1, f"Input shape mismatch: expected channel dimension 1, got {inp.shape[-3]}"
        esp = self.outSizePxl
        assert inp.shape[-2] == esp, f"Input shape mismatch: expected height {esp}, got {inp.shape[-2]}"
        assert inp.shape[-1] == esp, f"Input shape mismatch: expected width {esp}, got {inp.shape[-1]}"

    def pose_dic_from_pose(self, pose: np.ndarray):
        return {poseC: pose[..., i] for i, poseC in enumerate(self.poseCols)}

    def create_tmat_from_pose_dict(self, poseInfo: dict, prefix: str):
        """
        Args:
            poseInfo: dictionary with all required keys
            prefix: femur or tibia
        Returns:
            4x4 tensor
        """
        return torch.tensor(
            [
                [
                    poseInfo[f"{prefix}_rxx"],
                    poseInfo[f"{prefix}_rxy"],
                    poseInfo[f"{prefix}_rxz"],
                    poseInfo[f"{prefix}_tx"],
                ],
                [
                    poseInfo[f"{prefix}_ryx"],
                    poseInfo[f"{prefix}_ryy"],
                    poseInfo[f"{prefix}_ryz"],
                    poseInfo[f"{prefix}_ty"],
                ],
                [
                    poseInfo[f"{prefix}_rzx"],
                    poseInfo[f"{prefix}_rzy"],
                    poseInfo[f"{prefix}_rzz"],
                    poseInfo[f"{prefix}_tz"],
                ],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=self.DTYPE_TORCH,
        )

    def check_img_paths(self) -> (int, int):
        """check if every img_path exists on disk, returns number of bad and good paths"""
        b = 0
        g = 0
        for img_path in self.df["img_path"]:
            if not Path(img_path).is_file():
                logging.info("Img path {} was not found ...".format(img_path))
                b += 1
            else:
                g += 1
        return b, g