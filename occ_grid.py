
    # ___________________________________________ Occupancy Grid Related ________________________________________________
    @torch.no_grad()
    def update_occ_grid(self, global_step=0, occ_thre=0.1):
        def occ_eval_fn(x):  # 更新occupancy grid
            sdf_chunk = []
            color_chunk = []
            chunk = len(x)
            pbar = tqdm(range(0, len(x), chunk))
            for i in pbar:
                points = x[i:i + chunk]
                geo_out_chunk = self.sdf_network(points, with_grad=True, with_feature=True)
                sdf_chunk.append(geo_out_chunk[0])

                feature_chunk = geo_out_chunk[1]
                grad_chunk = geo_out_chunk[2]
                grad_chunk = grad_chunk / torch.norm(grad_chunk, dim=-1, keepdim=True)
                sampled_color = self.color_network(points, grad_chunk, -grad_chunk, feature_chunk)
                color_chunk.append(sampled_color)

            sdf = torch.cat(sdf_chunk, dim=0)
            color = torch.cat(color_chunk, dim=0)
            self.occ_color = color
            # sdf = self.sdf_network(x, with_grad=False, with_feature=False)[0]
            inv_s = self.deviation_network(torch.zeros([1, 3]))[:, :1].clip(1e-6, 1e6)
            inv_s = inv_s.expand(sdf.shape[0], 1)
            estimated_next_sdf = sdf[..., None] - self.render_step_size * 0.5
            estimated_prev_sdf = sdf[..., None] + self.render_step_size * 0.5
            prev_cdf = torch.sigmoid(estimated_prev_sdf * inv_s)
            next_cdf = torch.sigmoid(estimated_next_sdf * inv_s)
            p = prev_cdf - next_cdf
            c = prev_cdf
            alpha = ((p + 1e-5) / (c + 1e-5)).view(-1, 1).clip(0.0, 1.0)
            torch.cuda.empty_cache()
            return alpha

        torch.cuda.empty_cache()
        self.occupancy_grid.every_n_step(step=global_step, occ_eval_fn=occ_eval_fn,
                                         occ_thre=occ_thre)
        torch.cuda.empty_cache()

    def visual_occupancy_grid(self, show_color=True):
        def change_background_to_black(vis):
            opt = vis.get_render_option()
            opt.background_color = np.asarray([0, 0, 0])
            return False

        grid_indices = self.occupancy_grid.grid_coords
        valid_grid_mask = self.occupancy_grid.binary.view(-1, 1)[..., 0]
        valid_grid = grid_indices[valid_grid_mask]
        occ = self.occupancy_grid.occs[valid_grid_mask]

        valid_grid = valid_grid.detach().cpu().numpy()
        occ = occ.detach().cpu().numpy()

        color = self.occ_color[valid_grid_mask]
        color = color.detach().cpu().numpy()
        color_tmp = color

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(valid_grid)
        if show_color == False:
            color = (1 - occ[..., None]).repeat(3, axis=-1)  # np.zeros_like(occ[..., None]).repeat(3, axis=-1)
        pcd.colors = o3d.utility.Vector3dVector(color[:, :3])
        voxel_grid = o3d.geometry.VoxelGrid.create_from_point_cloud(pcd, voxel_size=1)

        o3d.visualization.draw_geometries_with_key_callbacks([voxel_grid], {ord("K"): change_background_to_black})


# 使用
        torch.cuda.empty_cache()
        self.render.update_occ_grid(0, occ_thre=0.1)
        self.render.visual_occupancy_grid()
        os.makedirs(os.path.join(self.base_exp_dir, 'checkpoints'), exist_ok=True)
        torch.save([self.render.occupancy_grid, self.render.occ_color],
                   os.path.join(self.base_exp_dir, 'checkpoints', f"occ_grid_{self.iter_step}.pth"))

