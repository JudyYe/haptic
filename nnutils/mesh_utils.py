# --------------------------------------------------------
# Written by Yufei Ye (https://github.com/JudyYe)
# --------------------------------------------------------
from __future__ import print_function
from pytorch3d.structures.meshes import join_meshes_as_batch
from typing import Union

import numpy as np
import pytorch3d.structures
import pytorch3d.structures.utils as struct_utils
import torch
from pytorch3d.renderer import (
    DirectionalLights,
    MeshRasterizer,
    PerspectiveCameras,
    RasterizationSettings,
    TexturesVertex,
    TexturesUV,
)
from pytorch3d.renderer.lighting import AmbientLights
from pytorch3d.renderer.mesh.shader import HardPhongShader
from pytorch3d.structures import Pointclouds
from pytorch3d.structures.meshes import Meshes
from pytorch3d.transforms import Scale, Transform3d, Translate

from . import geom_utils


def pad_texture(meshes: Meshes, feature: torch.Tensor = "white") -> TexturesVertex:
    """
    :param meshes:
    :param feature: (sumV, C)
    :return:
    """
    if isinstance(feature, TexturesVertex):
        return feature
    if feature == "white":
        feature = torch.ones_like(meshes.verts_padded())
    elif feature == "blue":
        feature = torch.zeros_like(meshes.verts_padded())
        # color = torch.FloatTensor([[[203,238,254]]]).to(meshes.device)  / 255
        color = (
            torch.FloatTensor([[[183, 216, 254]]]).to(meshes.device) / 255
        )  # * s - s/2
        feature = feature + color
    elif feature == "red":
        feature = torch.zeros_like(meshes.verts_padded())
        color = (
            torch.FloatTensor([[[254, 216 / 2, 183 / 2]]]).to(meshes.device) / 255
        )  # * s - s/2
        feature = feature + color
    elif feature == "pink":
        feature = torch.zeros_like(meshes.verts_padded())
        color = (
            torch.FloatTensor([[[255, 153, 204]]]).to(meshes.device) / 255
        )  # * s - s/2
        feature = feature + color
    elif feature == "yellow":
        feature = torch.zeros_like(meshes.verts_padded())
        # yellow = [250 / 255.0, 230 / 255.0, 154 / 255.0],
        color = torch.FloatTensor([[[240 / 255.0, 207 / 255.0, 192 / 255.0]]]).to(
            meshes.device
        )
        color = color * 2 - 1
        # color = torch.FloatTensor([[[250 / 255.0, 230 / 255.0, 154 / 255.0]]]).to(meshes.device) * 2 - 1
        feature = feature + color
    elif feature == "random":
        feature = torch.rand_like(meshes.verts_padded())  # [0, 1]
    if feature.dim() == 2:
        feature = struct_utils.packed_to_list(
            feature, meshes.num_verts_per_mesh().tolist()
        )
        # feature = struct_utils.list_to_padded(feature, pad_value=-1)

    texture = TexturesVertex(feature)
    texture._num_faces_per_mesh = meshes.num_faces_per_mesh().tolist()
    texture._num_verts_per_mesh = meshes.num_verts_per_mesh().tolist()
    texture._N = meshes._N
    texture.valid = meshes.valid
    return texture


def render_mesh(
    meshes: Meshes,
    cameras,
    rgb_mode=True,
    depth_mode=False,
    normal_mode=False,
    uv_mode=False,
    **kwargs,
):
    """
    flip issue: https://github.com/facebookresearch/pytorch3d/issues/78
    :param meshes:
    :param out_size: H=W
    :param cameras:
    :param kwargs:
    :return: 'rgb': (N, 3, H, W). 'mask': (N, 1, H, W). 'rgba': (N, 3, H, W)
    """
    image_size = kwargs.get("out_size", 224)
    raster_settings = kwargs.get(
        "raster_settings",
        RasterizationSettings(
            image_size=image_size,
            faces_per_pixel=2,
            bin_size=kwargs.get("bin_size", 0),
            cull_backfaces=kwargs.get("cull_backfaces", False),
        ),
    )
    device = cameras.device

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings).to(
        device
    )
    out = {}
    # import pdb; pdb.set_trace()
    fragments = rasterizer(meshes, **kwargs)
    out["frag"] = fragments

    if rgb_mode:
        shader = kwargs.get(
            "shader",
            HardPhongShader(
                device=meshes.device,
                lights=ambient_light(meshes.device, cameras, **kwargs),
            ),
        )
        image = shader(
            fragments, meshes, cameras=cameras, **kwargs
        )  # znear=znear, zfar=zfar, **kwargs)
        rgb, _ = flip_transpose_canvas(image)

        # get mask
        # Find out how much background_color needs to be expanded to be used for masked_scatter.
        N, H, W, K = fragments.pix_to_face.shape
        is_background = fragments.pix_to_face[..., 0] < 0  # (N, H, W)
        alpha = torch.ones((N, H, W, 1), dtype=rgb.dtype, device=device)
        alpha[is_background] = 0.0
        mask = flip_transpose_canvas(alpha, False)

        # Concat with the alpha channel.

        out["image"] = rgb
        out["mask"] = mask

    return out


def flip_transpose_canvas(image, rgba=True):
    image = torch.flip(image, dims=[1, 2])  # flip up-down, and left-right
    image = image.transpose(-1, -2).transpose(-2, -3)  # H, 4, W --> 4, H, W
    if rgba:
        rgb, mask = torch.split(image, [image.size(1) - 1, 1], dim=1)  # [0-1]
        return rgb, mask
    else:
        return image


def ambient_light(device="cpu", cameras: PerspectiveCameras = None, **kwargs):
    d = torch.FloatTensor([[0, 0, -1]]).to(device)
    N = 1 if cameras is None else len(cameras)
    zeros = torch.zeros([N, 3], device=device)
    d = zeros + d
    if cameras is not None:
        d = (
            cameras.get_world_to_view_transform()
            .inverse()
            .transform_normals(d.unsqueeze(1))
        )
        d = d.squeeze(1)

    color = kwargs.get("light_color", np.array([0.65, 0.3, 0.0]))
    D = kwargs.get("dims", 3)
    t_zeros = torch.zeros([N, D], device=device)
    if D == 3:
        am, df, sp = color
        am = t_zeros + am
        df = t_zeros + df
        sp = t_zeros + sp
        lights = DirectionalLights(
            device=device,
            ambient_color=am,
            diffuse_color=df,
            specular_color=sp,
            direction=d,
        )
    else:
        lights = MyAmbientLights(ambient_color=t_zeros + 1, device=device)
    return lights


class MyAmbientLights(AmbientLights):
    def __init__(self, *, ambient_color=None, device="cpu") -> None:
        super().__init__(ambient_color=ambient_color, device=device)
        self.D = self.ambient_color.shape[-1]

    def diffuse(self, normals, points) -> torch.Tensor:
        N = len(points)
        D = self.D
        return torch.zeros([N, D], device=self.device)

    def specular(self, normals, points, camera_position, shininess) -> torch.Tensor:
        N = len(points)
        D = self.D
        return torch.zeros([N, D], device=self.device)


# ### Transformation Utils ###
def apply_transform(geom: Union[Meshes, Pointclouds, torch.Tensor], trans: Transform3d):
    trans = Transform3d(matrix=trans.transpose(1, 2), device=trans.device)
    verts = get_verts(geom)
    dtype = verts.dtype
    verts = trans.transform_points(verts)
    verts = verts.to(dtype)  # for amp
    if hasattr(geom, "update_padded"):
        geom = geom.update_padded(verts)
    else:
        geom = verts
    return geom


def get_verts(geom: Union[Meshes, Pointclouds]) -> torch.Tensor:
    if isinstance(geom, Meshes) or isinstance(geom, pytorch3d.structures.Meshes):
        view_points = geom.verts_padded()
    elif isinstance(geom, Pointclouds):
        view_points = geom.points_padded()
    elif isinstance(geom, torch.Tensor):
        view_points = geom
    else:
        raise NotImplementedError(type(geom))
    return view_points


def get_nTw(geom: Meshes, new_center=None, new_scale=1):
    """get normalization transformation, that scale the geometry to [-1, 1]
    Args:
        geom (_type_): _description_
    """
    device = geom.device
    verts = get_verts(geom)  # (N, V, 3)
    # if verts are empty
    if isinstance(geom, Meshes) and geom.isempty():
        nTw = torch.eye(4, device=device)[None].repeat(len(geom), 1, 1)
        return nTw
    # get bounding bbox
    bbnx_max = torch.max(verts, dim=1, keepdim=False)[0]  # (N, 3)
    bbnx_min = torch.min(verts, dim=1, keepdim=False)[0]  # (N, 3)
    width, dim = torch.max(bbnx_max - bbnx_min, dim=-1, keepdim=False)  # (N,)
    empty_mask = (width == 0).float()
    width = empty_mask * 1 + (1 - empty_mask) * width
    # width = width.clamp(min=1e-5)

    if new_center is None:
        new_center = (bbnx_max + bbnx_min) / 2  # (N, 3)

    tsl = Translate(-new_center, device=device)
    scale = Scale(2 * new_scale / width, device=device)
    nTw = tsl.compose(scale)
    nTw = nTw.get_matrix().transpose(-1, -2)
    return nTw


def get_cTw_list(dist, view_mod, T=21, center=None, nTw=None):
    """get extrinsic matrix list

    Args:
        center (_type_): (N, 3)
        dist (_type_): _description_
    """
    assert center is not None or nTw is not None, "Specify either center or nTw"
    if nTw is None:
        N = len(center)
        device = center.device
        nTw = geom_utils.rt_to_homo(
            torch.eye(3, device=device)[None].repeat(N, 1, 1), -center
        )  # (N, 4, 4)

    device = nTw.device
    N = len(nTw)
    nTw_exp = nTw[None].repeat(T, 1, 1, 1).reshape(T * N, 4, 4)

    azel = get_view_list(view_mod, device, T)  # T, 2
    azel = azel.unsqueeze(1).repeat(1, N, 1).reshape(T * N, 2)
    R = geom_utils.azel_to_rot_v2(azel, homo=False)

    tsl = torch.zeros([T, N, 3], device=device)  # TODO
    tsl[..., 2] = dist
    tsl = tsl.reshape(T * N, 3)

    cTn = geom_utils.rt_to_homo(R, tsl)
    cTw = cTn @ nTw_exp

    cTw = cTw.reshape(T, N, 4, 4)
    return cTw


def render_geom_rot_v2(
    wGeom: Union[Meshes, Pointclouds],
    view_mod="az",
    scale_geom=True,
    view_centric=False,
    cameras: PerspectiveCameras = None,
    f=10,
    r=0.8,
    new_bound=1,
    time_len=21,
    nTw=None,
    **kwargs,
):
    geom = wGeom
    N = len(geom)
    device = geom.device
    if geom.isempty():
        H = kwargs.get("out_size", 224)
        return [torch.zeros([N, 3, H, H])]
    render = render_mesh
    if cameras is None:
        # ??????????
        cameras = PerspectiveCameras(f, device=device)
        if nTw is None:
            if scale_geom:
                nTw = get_nTw(geom, new_scale=new_bound)
            else:
                nTw = torch.eye(4)[None].repeat(N, 1, 1).to(device)
    # else:
    #     raise NotImplementedError('todo')

    dist = f * new_bound / r
    cTw_list = get_cTw_list(dist, view_mod, time_len, nTw=nTw)

    image_list = []
    for t in range(time_len):
        cTw = cTw_list[t]
        cGeom = apply_transform(wGeom, cTw)
        image = render(cGeom, cameras, **kwargs)
        image_list.append(image["image"])
    return image_list


def get_view_list(view_mod, device="cpu", time_len=21, **kwargs):
    """
    :param view_mod:
    :param x:
    :return: (T, 2) : azel, xyz
    """
    view_dict = {
        "az": (0, np.pi * 2, 10 / 180 * np.pi),
        "el": (-np.pi / 2, np.pi / 2, 5 / 180 * np.pi),
        "el0": (0, np.pi * 2, 10 / 180 * np.pi),
    }
    zeros = torch.zeros([time_len])
    if "az" in view_mod:
        vary_az = torch.linspace(
            view_dict[view_mod][0], view_dict[view_mod][1], time_len
        )
        vary_el = zeros
    elif "el" in view_mod:
        vary_az = zeros
        vary_el = torch.linspace(
            view_dict[view_mod][0], view_dict[view_mod][1], time_len
        )
    elif "circle" in view_mod:
        theta = torch.linspace(0, np.pi * 2, time_len)
        cone = 2 / 3
        vary_az = cone * torch.cos(theta)
        vary_el = cone * torch.sin(theta)
    else:
        raise NotImplementedError

    azel = torch.stack([vary_az, vary_el], dim=1).to(device)  # (T, 2)
    return azel




# ######## Batch utils ########
def join_scene(mesh_list) -> Meshes:
    """Joins a list of meshes to single Meshes of scene"""
    # simple check
    if len(mesh_list) == 1:
        return mesh_list[0]
    device = mesh_list[0].device

    v_list = []
    f_list = []
    t_list = []
    for m, mesh in enumerate(mesh_list):
        v_list.append(mesh.verts_list())
        f_list.append(mesh.faces_list())
        if mesh.textures is None or isinstance(mesh.textures, TexturesUV):
            mesh.textures = pad_texture(mesh)
        t_list.append(mesh.textures.verts_features_list())
        N = len(mesh)

    scene_list = []
    for n in range(N):
        verts = [v_list[m][n] for m in range(len(mesh_list))]
        faces = [f_list[m][n] for m in range(len(mesh_list))]
        texes = [t_list[m][n] for m in range(len(mesh_list))]
        # merge to one scene
        tex = TexturesVertex(texes)
        mesh = Meshes(verts, faces, textures=tex).to(device)
        tex = TexturesVertex(mesh.textures.verts_features_packed().unsqueeze(0))
        scene = Meshes(verts=mesh.verts_packed().unsqueeze(0), faces=mesh.faces_packed().unsqueeze(0), textures=tex)

        scene_list.append(scene)
    scene = join_meshes_as_batch(scene_list, include_textures=True)
    return scene