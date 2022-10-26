# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


import omni.usd
import omni.client

from pxr import UsdGeom, Sdf, UsdPhysics, UsdShade


# Note: this script should be executed in Isaac Sim `Script Editor` window

def create_ur10(asset_dir_usd_path, ur10_dir_usd_path):
    # Duplicate UR10 folder
    omni.client.copy(asset_dir_usd_path, ur10_dir_usd_path)

def create_ur10_mesh(asset_usd_path, ur10_mesh_usd_path):
    # Create ur10_mesh.usd file
    omni.client.copy(asset_usd_path, ur10_mesh_usd_path)
    omni.usd.get_context().open_stage(ur10_mesh_usd_path)
    stage = omni.usd.get_context().get_stage()
    edits = Sdf.BatchNamespaceEdit()
    # Create parent Xforms
    reparent_tasks = [
        # base_link
        ['/ur10/base_link/cylinder', 'geoms_xform'],
        ['/ur10/base_link/ur10_base', 'geoms_xform'],
        # shoulder_link
        ['/ur10/shoulder_link/cylinder', 'geoms_xform'],
        ['/ur10/shoulder_link/cylinder_0', 'geoms_xform'],
        ['/ur10/shoulder_link/ur10_shoulder', 'geoms_xform'],
        # upper_arm_link
        ['/ur10/upper_arm_link/cylinder', 'geoms_xform'],
        ['/ur10/upper_arm_link/cylinder_0', 'geoms_xform'],
        ['/ur10/upper_arm_link/cylinder_1', 'geoms_xform'],
        ['/ur10/upper_arm_link/ur10_upper_arm', 'geoms_xform'],
        # forearm_link
        ['/ur10/forearm_link/cylinder', 'geoms_xform'],
        ['/ur10/forearm_link/cylinder_0', 'geoms_xform'],
        ['/ur10/forearm_link/cylinder_1', 'geoms_xform'],
        ['/ur10/forearm_link/ur10_forearm', 'geoms_xform'],
        # wrist_1_link
        ['/ur10/wrist_1_link/cylinder', 'geoms_xform'],
        ['/ur10/wrist_1_link/cylinder_0', 'geoms_xform'],
        ['/ur10/wrist_1_link/ur10_wrist_1', 'geoms_xform'],
        # wrist_2_link
        ['/ur10/wrist_2_link/cylinder', 'geoms_xform'],
        ['/ur10/wrist_2_link/cylinder_0', 'geoms_xform'],
        ['/ur10/wrist_2_link/ur10_wrist_2', 'geoms_xform'],
        # wrist_3_link
        ['/ur10/wrist_3_link/cylinder', 'geoms_xform'],
        ['/ur10/wrist_3_link/ur10_wrist_3', 'geoms_xform'],
    ] # [prim_path, parent_xform_name]
    for task in reparent_tasks:
        prim_path, parent_xform_name = task
        old_parent_path = '/'.join(prim_path.split('/')[:-1])
        new_parent_path = f'{old_parent_path}/{parent_xform_name}'
        UsdGeom.Xform.Define(stage, new_parent_path)
        edits.Add(Sdf.NamespaceEdit.Reparent(prim_path, new_parent_path, -1))
    stage.GetRootLayer().Apply(edits)
    # Save to file
    omni.usd.get_context().save_stage()

def create_ur10_instanceable(ur10_mesh_usd_path, ur10_instanceable_usd_path):
    omni.client.copy(ur10_mesh_usd_path, ur10_instanceable_usd_path)
    omni.usd.get_context().open_stage(ur10_instanceable_usd_path)
    stage = omni.usd.get_context().get_stage()
    # Set up references and instanceables
    for prim in stage.Traverse():
        if prim.GetTypeName() != 'Xform':
            continue
        # Add reference to visuals_xform, collisions_xform, geoms_xform, and make them instanceable
        path = str(prim.GetPath())
        if path.endswith('visuals_xform') or path.endswith('collisions_xform') or path.endswith('geoms_xform'):
            ref = prim.GetReferences()
            ref.ClearReferences()
            ref.AddReference('./ur10_mesh.usd', path)
            prim.SetInstanceable(True)
    # Save to file
    omni.usd.get_context().save_stage()

def create_block_indicator():
    asset_usd_path = 'omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Props/Blocks/block.usd'
    block_usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2022.1/Isaac/Props/Blocks/block.usd'
    omni.client.copy(asset_usd_path, block_usd_path)
    omni.usd.get_context().open_stage(block_usd_path)
    stage = omni.usd.get_context().get_stage()
    edits = Sdf.BatchNamespaceEdit()
    edits.Add(Sdf.NamespaceEdit.Remove('/object/object/collisions'))
    stage.GetRootLayer().Apply(edits)
    omni.usd.get_context().save_stage()

    asset_usd_path = 'omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Props/Blocks/block_instanceable.usd'
    block_usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2022.1/Isaac/Props/Blocks/block_instanceable.usd'
    omni.client.copy(asset_usd_path, block_usd_path)
    omni.usd.get_context().open_stage(block_usd_path)
    stage = omni.usd.get_context().get_stage()
    edits = Sdf.BatchNamespaceEdit()
    edits.Add(Sdf.NamespaceEdit.Remove('/object/object/collisions'))
    stage.GetRootLayer().Apply(edits)
    omni.usd.get_context().save_stage()

if __name__ == '__main__':
    asset_dir_usd_path = 'omniverse://localhost/NVIDIA/Assets/Isaac/2022.1/Isaac/Robots/UR10'
    ur10_dir_usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2022.1/Isaac/Robots/UR10'
    ur10_usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2022.1/Isaac/Robots/UR10/ur10.usd'
    ur10_mesh_usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2022.1/Isaac/Robots/UR10/ur10_mesh.usd'
    ur10_instanceable_usd_path = 'omniverse://localhost/Projects/J3soon/Isaac/2022.1/Isaac/Robots/UR10/ur10_instanceable.usd'
    create_ur10(asset_dir_usd_path, ur10_dir_usd_path)
    create_ur10_mesh(ur10_usd_path, ur10_mesh_usd_path)
    create_ur10_instanceable(ur10_mesh_usd_path, ur10_instanceable_usd_path)
    create_block_indicator()
    print("Done!")
