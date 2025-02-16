import colorsys
import os.path as osp
from glob import glob

import cv2
import imageio
import numpy as np
import plotly.graph_objects as go
import torch
from matplotlib import cm
from matplotlib import pyplot as plt
from pytorch3d.renderer.cameras import PerspectiveCameras, look_at_view_transform
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import plot_scene

from jutils import geom_utils, mesh_utils


# visualizer
class Visualizer(object):
    def __init__(self) -> None:
        super().__init__()

    def vis_traj(self, traj_list, legend_list) -> go.Figure:
        """ """
        fig = go.Figure()
        for traj, legend in zip(traj_list, legend_list):
            traj = traj.detach().cpu().numpy()
            fig.add_trace(
                go.Scatter3d(
                    x=traj[:, 0],
                    y=traj[:, 1],
                    z=traj[:, 2],
                    mode="lines",
                    name=legend,
                )
            )
        return fig

    def cmp_one_wTraj(
        self, wTraj_gt, wTraj_pred, wHand_gt=None, wHand_pred=None
    ) -> go.Figure:
        """

        :param wTraj_gt: (T, 3)
        :param wTraj_pred: (T, 3)
        :param wHand_gt: hands in that only vis the 1st element
        :param wHand_pred: _description_
        """
        wTraj_pred = wTraj_pred.detach().cpu().numpy()
        wTraj_gt = wTraj_gt.detach().cpu().numpy()

        if wHand_gt is not None:
            gt_hand_t0 = Meshes(
                wHand_gt.verts_padded()[0:1], wHand_gt.faces_padded()[0:1]
            )
            pred_hand_t0 = Meshes(
                wHand_pred.verts_padded()[0:1], wHand_pred.faces_padded()[0:1]
            )
            gt_hand_t0.textures = mesh_utils.pad_texture(gt_hand_t0, "red")
            pred_hand_t0.textures = mesh_utils.pad_texture(pred_hand_t0, "blue")

            fig = plot_scene(
                {
                    "title": {
                        "gt": gt_hand_t0,
                        "pred": pred_hand_t0,
                    }
                }
            )
            verts_np = [
                wHand_gt.verts_packed().detach().cpu().numpy(),
                wHand_pred.verts_packed().detach().cpu().numpy(),
            ]
        else:
            fig = go.Figure()
            verts_np = []
        fig.add_trace(
            go.Scatter3d(
                x=wTraj_pred[:, 0],
                y=wTraj_pred[:, 1],
                z=wTraj_pred[:, 2],
                mode="lines",
                name="pred",
                line=dict(color="blue"),
            )
        )
        fig.add_trace(
            go.Scatter3d(
                x=wTraj_gt[:, 0],
                y=wTraj_gt[:, 1],
                z=wTraj_gt[:, 2],
                mode="lines",
                name="gt",
                line=dict(color="red"),
            )
        )
        # set xyz limit to minmax of wTraj_gt, wTraj_pred
        xyz = np.concatenate([wTraj_gt, wTraj_pred] + verts_np, axis=0)
        minmax = np.concatenate([xyz.min(0), xyz.max(0)], axis=0)
        minmax = minmax.reshape(2, 3)
        fig.update_layout(
            scene=dict(
                xaxis=dict(range=minmax[:, 0]),
                yaxis=dict(range=minmax[:, 1]),
                zaxis=dict(range=minmax[:, 2]),
            )
        )
        # equal axis
        fig.update_layout(scene=dict(aspectmode="cube"))

        return fig

    def _get_time_inds(self, F, timeslice):
        if timeslice.startswith("uni"):
            timeslice = int(timeslice.split("-")[1])
            time_inds = np.linspace(0, F - 1, timeslice).astype(int)
        elif timeslice.startswith("first"):
            t0, T, dt = timeslice.split("-")[1:]
            time_inds = np.arange(int(t0), int(T), int(dt))
            t_offset = F - 1 - time_inds[-1]
            time_inds = time_inds + t_offset
        else:
            time_inds = np.arange(0, F, timeslice)
        return time_inds

    def traj_in_cam(
        self,
        cObjs,
        B,
        F,
        cameras: PerspectiveCameras,
        img_size,
        canvas,
        timeslice="uni-10",
        alpha=1,
        decay_mode="exp-5",
        cJoints=None,
        cJoints2=None,
    ):
        """

        :param cObjs: _description_
        :param B: _description_
        :param F: _description_
        :param cameras: _description_
        :param img_size: _description_
        :param canvas: (B, 3, H, W?)
        """
        time_inds = self._get_time_inds(F, timeslice)

        W, H = img_size.split([1, 1], -1)
        W, H = W.squeeze(-1), H.squeeze(-1)
        H, W = H[0].item(), W[0].item()

        out = mesh_utils.render_mesh(cObjs, cameras, out_size=(H, W))
        images = out["image"].reshape(B, F, 3, H, W)[:, time_inds]
        mask = out["mask"].reshape(B, F, 1, H, W)[:, time_inds]
        image = self.overlay_fainted_mask(
            images, mask, canvas, alpha=alpha, decay_mode=decay_mode
        )

        if cJoints is not None:
            J = cJoints.shape[-2]
            cJoints = cJoints.reshape(B * F, J, 3)
            iJoints = cameras.transform_points(cJoints.reshape(B * F, J, 3))[..., :2]
            print("cjoints", iJoints.shape, cJoints.shape)
            iJoints = iJoints.reshape(B, F, J, 2)
            iJoints = iJoints.detach().cpu().numpy()
            # draw traj
            image = self.draw_root(image, iJoints)
        if cJoints2 is not None:
            J = cJoints2.shape[-2]
            cJoints2 = cJoints2.reshape(B * F, J, 3)
            iJoints2 = cameras.transform_points(cJoints2.reshape(B * F, J, 3))[..., :2]
            iJoints2 = iJoints2.reshape(B, F, J, 2)
            iJoints2 = iJoints2.detach().cpu().numpy()
            image = self.draw_root(image, iJoints2)
        return image

    def track_in_cam(self, cObjs, B, F, cameras: PerspectiveCameras, img_size, canvas):
        W, H = img_size.split([1, 1], -1)
        W, H = W.squeeze(-1), H.squeeze(-1)

        intr = cameras.get_projection_transform().get_matrix().transpose(-1, -2)
        # fx, fy = intr[..., 0, 0], intr[..., 1, 1]
        # px, py = intr[..., 0, 2], intr[..., 1, 2]
        cameras = self.get_cameras(intr, H, W)
        iPoints = cameras.transform_points(cObjs.verts_padded())[..., :2]  # (B*F, V, 3)
        # iPoints = (iPoints / 2  + 0.5) * img_size.unsqueeze(1)
        iPoints = iPoints.reshape(B, F, -1, 2)  # (B, F, V, 2)
        iPoints = iPoints.detach().cpu().numpy()

        H, W = H[0].item(), W[0].item()

        canvas = canvas.reshape(B, F, 3, H, W)
        canvas_np = (
            canvas.permute(0, 1, 3, 4, 2).detach().cpu().numpy() / 2 + 0.5
        ) * 255
        canvas_np = np.clip(canvas_np, 0, 255).astype(np.uint8)

        trail_list = []
        for b in range(B):
            img_list_np = canvas_np[b]
            trail = vis_trail(
                img_list_np,
                iPoints[b],
            )
            trail_list.append(trail)

        trail = np.stack(trail_list)
        # B, F, 3, H, W -> F, B, 3, H, W
        trail = torch.FloatTensor(trail).permute(1, 0, 4, 2, 3) / 255 * 2 - 1
        return trail

    def root_in_world(self, wPoints, B, F, H=512, f=10, canvas=None):
        device = wPoints.device

        verts = wPoints
        verts = verts.reshape(B, -1, 3)

        # lookat
        cTw, cameras = get_lookat_cameras(verts, f, device=device)  # (B, 4, 4)
        cTw_exp = cTw[:, None].repeat(1, F, 1, 1).reshape(B * F, 4, 4)
        nPoints = mesh_utils.apply_transform(
            verts.reshape(B * F, -1, 3), cTw_exp
        )  # (BF, V, 3)
        iPoints = cameras.transform_points_ndc(nPoints)[..., :2]
        iPoints = (iPoints / 2 + 0.5) * H
        iPoints = iPoints.reshape(B, F, -1, 2)
        iPoints = iPoints.detach().cpu().numpy()

        canvas = canvas.reshape(B, 1, 3, H, H).repeat(1, F, 1, 1, 1)
        canvas_np = (
            canvas.permute(0, 1, 3, 4, 2).detach().cpu().numpy() / 2 + 0.5
        ) * 255
        canvas_np = np.clip(canvas_np, 0, 255).astype(np.uint8)

        trail_list = []
        for b in range(B):
            img_list_np = canvas_np[b]
            trail = vis_trail(
                img_list_np,
                iPoints[b],
            )
            trail_list.append(trail)

        trail = np.stack(trail_list)
        # B, F, 3, H, W -> F, B, 3, H, W
        trail = torch.FloatTensor(trail).permute(1, 0, 4, 2, 3) / 255 * 2 - 1
        return trail

    def root_in_cam(self, cPoints, B, F, cameras, img_size, canvas):
        W, H = img_size.split([1, 1], -1)
        W, H = W.squeeze(-1), H.squeeze(-1)

        intr = cameras.get_projection_transform().get_matrix().transpose(-1, -2)
        cameras = self.get_cameras(intr, H, W)
        iPoints = cameras.transform_points(cPoints)[..., :2]  # (B*F, V, 3)
        iPoints = iPoints.reshape(B, F, -1, 2)  # (B, F, V, 2)
        iPoints = iPoints.detach().cpu().numpy()

        H, W = H[0].item(), W[0].item()

        # canvas_np = (canvas.reshape(B, 3, H, W).detach().cpu().numpy() / 2 + 0.5) * 255
        canvas = canvas.reshape(B, 1, 3, H, W).repeat(1, F, 1, 1, 1)
        canvas_np = (
            canvas.permute(0, 1, 3, 4, 2).detach().cpu().numpy() / 2 + 0.5
        ) * 255
        canvas_np = np.clip(canvas_np, 0, 255).astype(np.uint8)

        trail_list = []
        for b in range(B):
            img_list_np = canvas_np[b : b + 1]
            trail = vis_trail(
                img_list_np,
                iPoints[b],
            )
            trail_list.append(trail)

        trail = np.stack(trail_list)
        # B, F, 3, H, W -> F, B, 3, H, W
        trail = torch.FloatTensor(trail).permute(1, 0, 4, 2, 3) / 255 * 2 - 1
        return trail

    def traj_in_world(
        self,
        wObjs,
        B,
        F,
        H=512,
        nTw=None,
        f=10,
        timeslice="uni-10",
        alpha=1,
        decay_mode="exp-5",
        wJoints=None,
        wJoints2=None,
    ):
        time_inds = self._get_time_inds(F, timeslice)

        device = wObjs.device

        T = len(time_inds)
        verts = wObjs.verts_padded()  # (B*F, V, 3)
        verts = verts.reshape(B, -1, 3)

        # lookat
        cTw, cameras = get_lookat_cameras(verts, f, device=device)  # (B, 4, 4)
        cTw_exp = cTw[:, None].repeat(1, F, 1, 1).reshape(B * F, 4, 4)

        cObjs = mesh_utils.apply_transform(
            wObjs,
            cTw_exp,
        )
        canvas = mesh_utils.render_mesh(cObjs, cameras, out_size=(H, H))

        image = canvas["image"].reshape(B, F, 3, H, H)[:, time_inds]
        mask = canvas["mask"].reshape(B, F, 1, H, H)[:, time_inds]
        bg = torch.ones_like(image[:, 0])
        overlayed = self.overlay_fainted_mask(
            image, mask, bg, decay_mode=decay_mode, alpha=alpha
        )

        if wJoints is not None:
            J = wJoints.shape[-2]
            wJoints = wJoints.reshape(B * F, J, 3)
            cJoints = mesh_utils.apply_transform(
                wJoints,
                cTw_exp,
            )
            iJoints = cameras.transform_points(cJoints.reshape(B, F * J, 3))[..., :2]
            # print(cJoints, iJoints)
            iJoints = (iJoints / 2 + 0.5) * H
            iJoints = iJoints.reshape(B, F, J, 2)
            iJoints = iJoints.detach().cpu().numpy()
            # draw traj
            overlayed = self.draw_root(overlayed, iJoints)
        if wJoints2 is not None:
            print("not none")
            J = wJoints2.shape[-2]
            wJoints2 = wJoints2.reshape(B * F, J, 3)
            cJoints2 = mesh_utils.apply_transform(
                wJoints2,
                cTw_exp,
            )
            iJoints2 = cameras.transform_points(cJoints2.reshape(B, F * J, 3))[..., :2]
            # print(cJoints, iJoints)
            iJoints2 = (iJoints2 / 2 + 0.5) * H
            iJoints2 = iJoints2.reshape(B, F, J, 2)
            iJoints2 = iJoints2.detach().cpu().numpy()
            # draw traj
            overlayed = self.draw_root(overlayed, iJoints2)
        return overlayed

    def draw_root(self, images, iJoints):
        """

        :param images: (B, C, H, W)
        :param iJoints: (B, F, J, 2)
        """
        image_np = (images.permute(0, 2, 3, 1).detach().cpu().numpy() / 2 + 0.5) * 255
        image_np = np.clip(image_np, 0, 255).astype(np.uint8)
        B, F, J = iJoints.shape[:3]
        for b in range(B):
            # draw lines
            for f in range(1, F):
                for j in range(J):
                    x1, y1 = iJoints[b, f - 1, j]
                    x2, y2 = iJoints[b, f, j]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    cv2.line(
                        image_np[b], (x1, y1), (x2, y2), (254, 216 / 2, 183 / 2), 2
                    )
        out = torch.FloatTensor(image_np).permute(0, 3, 1, 2) / 255 * 2 - 1
        return out

    def _get_decay_alpha(self, t, T, alpha, decay_mode):
        if decay_mode == "linear":
            alpha_t = alpha * ((t + 1) / T)
        elif decay_mode.startswith("exp"):
            sigma = float(decay_mode.split("-")[1])
            tt = (t + 1) / T  # (0, 1)
            alpha_t = alpha * (np.exp(sigma * tt - sigma))  # / (np.exp(1) - 1)
        alpha_t = max(0.25, alpha_t)
        alpha_t = min(1, alpha_t)
        return alpha_t

    def overlay_fainted_mask(self, images, masks, bg, alpha=1, decay_mode="exp-5"):
        """

        :param images: (B, T, 3, H, W)
        :param masks: (B, T, 3, H, W)
        :param bg: (B, 3, H, W)
        :param r: time decay defaults to 0.5
        """
        # overlay image on mask, with fainted masks
        B, T = images.shape[:2]
        canvas = bg
        for t in range(T):
            alpha_t = self._get_decay_alpha(t, T, alpha, decay_mode)
            canvas = blend(images[:, t], canvas, masks[:, t], alpha_t)
        return canvas

    def track_in_world(self, wObjs, B, F, H=512, nTw=None, f=10):
        device = wObjs.device

        verts = wObjs.verts_padded()  # (B*F, V, 3)
        verts = verts.reshape(B, -1, 3)

        # lookat
        cTw, cameras = get_lookat_cameras(verts, f, device=device)  # (B, 4, 4)
        cTw_exp = cTw[:, None].repeat(1, F, 1, 1).reshape(B * F, 4, 4)
        nPoints = mesh_utils.apply_transform(
            verts.reshape(B * F, -1, 3), cTw_exp
        )  # (BF, V, 3)
        iPoints = cameras.transform_points_ndc(nPoints)[..., :2]
        iPoints = (iPoints / 2 + 0.5) * H
        iPoints = iPoints.reshape(B, F, -1, 2)
        iPoints = iPoints.detach().cpu().numpy()

        cObjs = mesh_utils.apply_transform(
            wObjs,
            cTw_exp,
        )
        canvas = mesh_utils.render_mesh(cObjs, cameras, out_size=(H, H))["image"]

        canvas = canvas.reshape(B, F, 3, H, H)
        canvas_np = (
            canvas.permute(0, 1, 3, 4, 2).detach().cpu().numpy() / 2 + 0.5
        ) * 255
        canvas_np = np.clip(canvas_np, 0, 255).astype(np.uint8)

        trail_list = []
        for b in range(B):
            img_list_np = canvas_np[b]
            trail = vis_trail(
                img_list_np,
                iPoints[b],
            )
            trail_list.append(trail)

        trail = np.stack(trail_list)
        # B, F, 3, H, W -> F, B, 3, H, W
        trail = torch.FloatTensor(trail).permute(1, 0, 4, 2, 3) / 255 * 2 - 1
        return trail, (nTw, cameras)

    def vis_in_cam(self, cObjs, B, F, cameras, img_size, bg=None):
        N = len(cObjs)
        device = cObjs.device

        verts = cObjs.verts_padded()  # (B*F, V, 3)
        verts = verts.reshape(B, -1, 3)

        W, H = img_size.split([1, 1], -1)
        W, H = W.squeeze(-1), H.squeeze(-1)

        iHand = mesh_utils.render_mesh(
            cObjs, cameras, out_size=(H[0].item(), W[0].item())
        )
        out = blend(iHand["image"], bg, iHand["mask"])
        out = out.reshape(B, F, 3, H[0].item(), W[0].item()).transpose(0, 1)
        return out

    def vis_in_world(self, wObjs, B, F, H=512, nTw=None, f=10):
        """
        :return: images in shape of (T, B, 3, H, H)
        :return: nTw in shape of (B, 4, 4)
        """
        N = len(wObjs)
        device = wObjs.device
        # coord = plot_utils.create_coord(device, N, 0.1)

        verts = wObjs.verts_padded()  # (B*F, V, 3)
        verts = verts.reshape(B, -1, 3)
        # wObjs = mesh_utils.join_scene([wObjs, coord])

        if nTw is None:
            nTw = mesh_utils.get_nTw(verts)  # (B, 4, 4)
        nTw_exp = nTw[:, None].repeat(1, F, 1, 1).reshape(B * F, 4, 4)
        cameras = PerspectiveCameras(f, device=device)

        # nObjs = mesh_utils.apply_transform(wObjs, nTw)
        image_list = mesh_utils.render_geom_rot_v2(
            wObjs, time_len=1, nTw=nTw_exp, out_size=(H, H), cameras=cameras
        )
        canvas = image_list[0]
        canvas = canvas.reshape(B, F, 3, H, H).transpose(0, 1)  # (T, B, 3, H, W)
        return canvas, (nTw, cameras)

    def vis_in_ct(self, cTw, wObjs, cameras, H, W, images=None, r=0.9):
        B, T = cTw.shape[:2]

        cObjs = mesh_utils.apply_transform(
            wObjs,
            cTw.reshape(-1, 4, 4),
        )

        fgs = mesh_utils.render_mesh(cObjs, cameras, out_size=(H, W))
        if images is not None:
            images = images.reshape(B * T, 3, H, W)
            canvas = (
                fgs["mask"] * (r * fgs["image"] + (1 - r) * images)
                + (1 - fgs["mask"]) * images
            )
        else:
            canvas = fgs["image"]
        canvas = canvas.reshape(B, T, 3, H, W).transpose(0, 1)
        return canvas

    def vis_in_c1(self, cTw, wObjs, cameras, H, W, images=None, r=0.9):
        """

        :param cTw: (B, T, 4, 4)
        :param wObjs: (B, T, ...)
        :param cameras
        :param out_size:
        :param images: (B, T, 3, H, W)
        :return images in shape of (T, B, 3, H, W)
        """
        B, T = cTw.shape[:2]
        c1Tw = cTw[:, 0:1].repeat(1, T, 1, 1).reshape(B * T, 4, 4)

        cObjs = mesh_utils.apply_transform(
            wObjs,
            c1Tw,
        )

        fgs = mesh_utils.render_mesh(cObjs, cameras, out_size=(H, W))
        if images is not None:
            canvas = (
                fgs["mask"] * (r * fgs["image"] + (1 - r) * images)
                + (1 - fgs["mask"]) * images
            )
        else:
            canvas = fgs["image"]
        canvas = canvas.reshape(B, T, 3, H, W).transpose(0, 1)
        return canvas

    @staticmethod
    def get_cameras(intr, H, W, pt3d=True):
        """

        :param intr: pixel intrinsics (B, F, 3, 3)
        :param H: _description_
        :param W: _description_
        return PerspectiveCameras in batches (B*F)
        """
        device = intr.device
        flat_dim = int(np.prod(intr.shape[:-2]))
        # B, F = intr.shape[:2]
        fxfy = torch.stack([intr[..., 0, 0], intr[..., 1, 1]], dim=-1)  # (B, F, 2)
        pxpy = torch.stack([intr[..., 0, 2], intr[..., 1, 2]], dim=-1)
        if torch.is_tensor(H):
            HW = torch.stack([H, W], dim=-1)
        else:
            HW = torch.tensor([[H, W]], device=device).repeat(flat_dim, 1)
        fxfy = fxfy.reshape(flat_dim, 2)
        pxpy = pxpy.reshape(flat_dim, 2)
        # HW = HW.reshape(B*F, 2)

        # WEIRD SCREEN SPACE IN PYTORCH3D
        if pt3d:
            pxpy[..., 0] = W - pxpy[..., 0]
            pxpy[..., 1] = H - pxpy[..., 1]
        cameras = PerspectiveCameras(
            fxfy, pxpy, in_ndc=False, image_size=HW, device=device
        )
        return cameras

    def draw_kp2d(self, images, kp2d, color=(0, 255, 0), size=5):
        """
        :param images: (B, T 3, H, W)
        :param kp2d: (B, T J*2)
        :return: images in shape of (T, B 3, H, W)
        """
        B, T = kp2d.shape[:2]
        H, W = images.shape[-2:]
        canvas = (images.cpu().permute(0, 1, 3, 4, 2).detach().numpy() / 2 + 0.5) * 255
        canvas = np.clip(canvas, 0, 255).astype(np.uint8).copy()
        kp2d = kp2d.cpu().detach().numpy().reshape(B, T, -1, 2)
        kp2d[..., 0] = kp2d[..., 0] * W / 2 + W / 2
        kp2d[..., 1] = kp2d[..., 1] * H / 2 + H / 2
        for b in range(B):
            for t in range(T):
                for j in range(kp2d.shape[-2]):
                    x, y = kp2d[b, t, j]
                    x, y = int(x), int(y)
                    cv2.circle(canvas[b, t], (x, y), size, color, -1)

        canvas_tensor = torch.FloatTensor(canvas).permute(1, 0, 4, 2, 3) / 255 * 2 - 1
        return canvas_tensor

    def plt_depths(self, depth_list, legend_list) -> plt.Figure:
        """
        draw B subplot, each draws depth-T curves
        :param depth_list: [(B, T, ), ... ]
        :param legend_list: [str, ...]
        """
        B = depth_list[0].shape[0]
        plt.close()
        if B == 1:
            fig, axs = plt.subplots(1, 1, figsize=(10, 5))
            for depth, legend in zip(depth_list, legend_list):
                depth = depth.detach().cpu().numpy()
                axs.plot(depth[0], label=legend)
            axs.legend()
        else:
            fig, axs = plt.subplots(B, 1, figsize=(10, 5 * B))
            for b in range(B):
                for depth, legend in zip(depth_list, legend_list):
                    depth = depth.detach().cpu().numpy()
                    axs[b].plot(depth[b], label=legend)
                axs[b].legend()
        return fig


color_map = cm.get_cmap("jet")


def vis_trail(
    images, kpts_foreground, kpts_background=None, save_path=None, img_dir=None
):
    """
    This function calculates the median motion of the background, which is subsequently
    subtracted from the foreground motion. This subtraction process "stabilizes" the camera and
    improves the interpretability of the foreground motion trails.
    :param keypoints: (T, P, 2) array of keypoints
    """
    if images is None:
        img_files = sorted(list(glob(osp.join(img_dir, "*.png"))))
        images = np.array([imageio.imread(img_file) for img_file in img_files])

    NP = 10
    dp = max(kpts_foreground.shape[1] // NP, 1)
    kpts_foreground = kpts_foreground[:, 0::dp]  # can adjust kpts sampling rate here

    num_imgs, num_pts = kpts_foreground.shape[:2]

    frames = []

    for i in range(num_imgs):
        kpts = kpts_foreground  # - np.median(kpts_background - kpts_background[i], axis=1, keepdims=True)
        img_curr = images[i]

        for t in range(i):
            img1 = img_curr.copy()

            for j in range(num_pts):
                color = np.array(color_map(j / max(1, float(num_pts - 1)))[:3]) * 255

                color_alpha = 0.4

                hsv = colorsys.rgb_to_hsv(color[0], color[1], color[2])
                color = colorsys.hsv_to_rgb(hsv[0], hsv[1] * color_alpha, hsv[2])

                pt1 = kpts[t, j]
                pt2 = kpts[t + 1, j]
                p1 = (int(round(pt1[0])), int(round(pt1[1])))
                p2 = (int(round(pt2[0])), int(round(pt2[1])))

                cv2.line(img1, p1, p2, color, thickness=2)

            alpha = max(1 - 0.9 * ((i - t) / ((i + 1) * 0.99)), 0)
            # win_size = 100
            # alpha = (t + 1) / win_size
            img_curr = cv2.addWeighted(img1, alpha, img_curr, 1 - alpha, 0)

        for j in range(num_pts):
            color = np.array(color_map(j / max(1, float(num_pts - 1)))[:3]) * 255
            pt1 = kpts[i, j]
            p1 = (int(round(pt1[0])), int(round(pt1[1])))
            # cv2.circle(img_curr, p1, 1, color, -1, lineType=16)

        frames.append(img_curr)

    # imageio.mimwrite(save_path, frames, quality=8, fps=10)
    # imageio.imwrite(save_path.replace(".mp4", ".png"), frames[-1])
    return frames


def get_lookat_cameras(geom, focal=10, device="cuda:0", dist=0.75, min_max_size=None):
    points = mesh_utils.get_verts(geom)  # (N, V, 3)
    center = points.mean(1)  # (N, 3, )
    max_size = points.max(1)[0] - points.min(1)[0]  # (N, 3)
    max_size = max_size.max(-1)[0]  # max of x, y, z
    if min_max_size is not None:
        print(max_size)
        max_size = max_size.clamp(min=min_max_size)
    R, T = look_at_view_transform(
        dist=focal * max_size * dist, elev=0, azim=0, at=center, device=device
    )  # world to view
    cTw = geom_utils.rt_to_homo(R.transpose(-1, -2), T)
    cameras = PerspectiveCameras(focal_length=focal).to(device)

    # cameras = PerspectiveCameras(focal_length=focal, R=R, T=T).to(device)
    return cTw, cameras


def blend(fg, bg, mask, r=0.9):
    fg = fg.cpu()
    bg = bg.cpu()
    mask = mask.cpu()
    return mask * (fg * r + (1 - r) * bg) + (1 - mask) * bg
