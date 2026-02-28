import os
import argparse
import torch
import trimesh
import numpy as np
from pyhocon import ConfigFactory
from models.dataset import Dataset
from models.fields import RenderingNetwork, SDFNetwork, SingleVarianceNetwork, NeRF
from models.renderer import NeuSRenderer

class ColoredMeshExtractor:
    def __init__(self, conf_path, case='CASE_NAME', checkpoint=None):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load configuration
        f = open(conf_path)
        conf_text = f.read()
        conf_text = conf_text.replace('CASE_NAME', case)
        f.close()

        self.conf = ConfigFactory.parse_string(conf_text)
        self.conf['dataset.data_dir'] = self.conf['dataset.data_dir'].replace('CASE_NAME', case)
        self.base_exp_dir = self.conf['general.base_exp_dir']
        self.dataset = Dataset(self.conf['dataset'])

        # Create networks
        self.nerf_outside = NeRF(**self.conf['model.nerf']).to(self.device)
        self.sdf_network = SDFNetwork(**self.conf['model.sdf_network']).to(self.device)
        self.deviation_network = SingleVarianceNetwork(**self.conf['model.variance_network']).to(self.device)
        self.color_network = RenderingNetwork(**self.conf['model.rendering_network']).to(self.device)

        self.renderer = NeuSRenderer(self.nerf_outside,
                                    self.sdf_network,
                                    self.deviation_network,
                                    self.color_network,
                                    **self.conf['model.neus_renderer'])

        # Load checkpoint
        if checkpoint is not None:
            self.load_checkpoint(checkpoint)

        # Set all networks to evaluation mode
        self.nerf_outside.eval()
        self.sdf_network.eval()
        self.deviation_network.eval()
        self.color_network.eval()

    def load_checkpoint(self, checkpoint):
        checkpoint_path = os.path.join(self.base_exp_dir, 'checkpoints', checkpoint)
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.nerf_outside.load_state_dict(checkpoint['nerf'])
        self.sdf_network.load_state_dict(checkpoint['sdf_network_fine'])
        self.deviation_network.load_state_dict(checkpoint['variance_network_fine'])
        self.color_network.load_state_dict(checkpoint['color_network_fine'])

    def extract_colored_mesh(self, resolution=512, threshold=0.0):
        bound_min = torch.tensor(self.dataset.object_bbox_min, dtype=torch.float32).to(self.device)
        bound_max = torch.tensor(self.dataset.object_bbox_max, dtype=torch.float32).to(self.device)

        print(f"Extracting geometry with resolution {resolution} and threshold {threshold}")
        vertices, triangles = self.renderer.extract_geometry(bound_min, bound_max, resolution=resolution, threshold=threshold)
        print(f"Extracted {len(vertices)} vertices and {len(triangles)} triangles")

        # Calculate colors for vertices
        vertices_color = []
        batch_size = 1000  # Process vertices in batches

        for i in range(0, len(vertices), batch_size):
            vertex_batch = vertices[i:i+batch_size]
            pts = torch.tensor(vertex_batch, dtype=torch.float32).to(self.device)
            view_dirs = torch.zeros_like(pts)  # Assuming view direction doesn't matter for color

            with torch.enable_grad():  # Temporarily enable gradients
                pts.requires_grad_(True)
                sdf_nn_output = self.sdf_network(pts)
                feature_vectors = sdf_nn_output[:, 1:]
                sdf = sdf_nn_output[:, :1]
                gradients = torch.autograd.grad(sdf, pts, torch.ones_like(sdf), create_graph=True)[0]

            with torch.no_grad():  # Disable gradients for color computation
                colors = self.color_network(pts, gradients, view_dirs, feature_vectors)
                colors = torch.sigmoid(colors)  # Apply sigmoid to get values in [0, 1]
                colors = (colors.cpu().numpy() * 255).astype(np.uint8)  # Convert to uint8

            vertices_color.extend(colors)

            if i % 10000 == 0:
                print(f"Processed {i}/{len(vertices)} vertices")

        vertices_color = np.array(vertices_color)

        print(f"Color array shape: {vertices_color.shape}")
        if len(vertices_color) > 0:
            print(f"Color value range: min={vertices_color.min(axis=0)}, max={vertices_color.max(axis=0)}")
        else:
            print("No colors were computed.")

        # Create mesh
        mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=vertices_color)

        # Save mesh
        os.makedirs(os.path.join(self.base_exp_dir, 'colored_meshes'), exist_ok=True)
        output_path = os.path.join(self.base_exp_dir, 'colored_meshes', 'colored_mesh.ply')
        mesh.export(output_path)
        print(f"Colored mesh saved to {output_path}")

        # Save a simple colored mesh for comparison
        simple_colors = np.full_like(vertices_color, [255, 0, 0])  # Red color
        simple_mesh = trimesh.Trimesh(vertices, triangles, vertex_colors=simple_colors)
        simple_output_path = os.path.join(self.base_exp_dir, 'colored_meshes', 'simple_colored_mesh.ply')
        simple_mesh.export(simple_output_path)
        print(f"Simple colored mesh saved to {simple_output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--conf', type=str, required=True)
    parser.add_argument('--case', type=str, required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--threshold', type=float, default=0.0)
    args = parser.parse_args()

    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    extractor = ColoredMeshExtractor(args.conf, args.case, args.checkpoint)
    extractor.extract_colored_mesh(resolution=args.resolution, threshold=args.threshold)

if __name__ == "__main__":
    main()  