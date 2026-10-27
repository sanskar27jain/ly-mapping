import tables as tab
import numpy as np

DETECTOR_BOUNDS = {
    "x": (-400.0, 400.0),
    "y": (-425.0, 425.0),
    "z": (-300.0, 600.0),
} 

reco_track_point_bounds = DETECTOR_BOUNDS.copy()
reco_track_point_bounds['x'] = (-345, 345) 
# The (lardon z) (larsoft x) values are computed from e drift times referenced to estimated t0 (actual track-crossing time). 
# It should be impossible to get a (larsoft) x significantly outside active volume unless t0 estimation is wrong (if the first e- reached the anode "before the track even crossed"). 

ACTIVE_VOLUME_BOUNDS = {
    "x": (-338.5, 338.5),
    "y": (-334.0, 334.0),
    "z": (0,      300.0),
}  


def get_trajectory_start_end(trajectory_table_row : tab.tableextension.Row, exclude_interior_tracks=-1, exclude_y=-1):
    """Get the start and end points of each trajectory from the read_lardon_tracks hdf5 table"""

    active_v_lower_bounds = np.array([av_bounds[0] for ax, av_bounds in ACTIVE_VOLUME_BOUNDS.items()])
    active_v_upper_bounds = np.array([av_bounds[1] for ax, av_bounds in ACTIVE_VOLUME_BOUNDS.items()])
    inside_xyz_bounds_by_margin = lambda point, margin : ((point - active_v_lower_bounds) > margin) & ((active_v_upper_bounds - point) > margin) 

    # LArSoft (x,y,z) <---> LARDON (z,x,y)
    coords = {}
    coords["x_ini"], coords["y_ini"], coords["z_ini"] = trajectory_table_row['z_ini_corrected'], trajectory_table_row['x_ini'], trajectory_table_row['y_ini']  
    coords["x_end"], coords["y_end"], coords["z_end"] = trajectory_table_row['z_end_corrected'], trajectory_table_row['x_end'], trajectory_table_row['y_end']  

    is_paired = (trajectory_table_row["cathode_crosser_ID"] != -1)
    if is_paired:
        # The "cc_"-labelled tracks are supposed to be the ones in the upper volume.  
        coords["cc_x_ini"], coords["cc_y_ini"], coords["cc_z_ini"] = trajectory_table_row['cc_z_ini_corrected'], trajectory_table_row['cc_x_ini'], trajectory_table_row['cc_y_ini'] 
        coords["cc_x_end"], coords["cc_y_end"], coords["cc_z_end"] = trajectory_table_row['cc_z_end_corrected'], trajectory_table_row['cc_x_end'], trajectory_table_row['cc_y_end'] 

        ## Perform "quality control" tests
        # If any of the four points lie outside the allowed bounds, exit.
        for ax, bounds in reco_track_point_bounds.items():
            if np.any( [ ((coords[k] < bounds[0]) or (coords[k] > bounds[1])) for k in (f"cc_{ax}_ini",f"cc_{ax}_end",f"{ax}_ini",f"{ax}_end") ] ):
                return None

        # If the "cc_" (upper volume) track starts _below_ the start of the paired *lower* volume track (or ends below its end), something is wrong.
        # Basically if one track completely overlaps vertically with the other or the tracks are "on the wrong side", move on.    
        if (coords["cc_x_ini"] < coords["x_ini"]) or (coords["cc_x_end"] < coords["x_end"]):
            return None
        # Note: at this point the cc_ini --> end distance ought to be the largest among the four points involved. Can check this externally 
        start = np.array([ coords['cc_x_ini'], coords['cc_y_ini'], coords['cc_z_ini'] ])
        end   = np.array([ coords['x_end'],    coords['y_end'],    coords['z_end'] ])
        # return start, end
    else:
        # If any of the four points lie outside the allowed bounds, exit.
        for ax, bounds in reco_track_point_bounds.items():
            if np.any( [ ((coords[k] < bounds[0]) or (coords[k] > bounds[1])) for k in (f"{ax}_ini",f"{ax}_end") ] ):
                return None

        start = np.array([ coords['x_ini'], coords['y_ini'], coords['z_ini'] ])
        end   = np.array([ coords['x_end'], coords['y_end'], coords['z_end'] ])

    if exclude_interior_tracks < 0 and exclude_y < 0:
        return start, end
    else:
        if exclude_interior_tracks > 0:
            start_in_av_interior = np.all(inside_xyz_bounds_by_margin(start, exclude_interior_tracks))
            end_in_av_interior   = np.all(inside_xyz_bounds_by_margin(end, exclude_interior_tracks))  
            if start_in_av_interior or end_in_av_interior:
                return None 
            
        if exclude_y > 0:
            if (np.abs(start[1]) <= exclude_y) or (np.abs(end[1]) <= exclude_y):
                return None
        
    return start, end


def project_segment_to_detector_bounds(start, end, flipped=False):
    """Project a trajectory's start and end points to the detector bounds""" 
    start, end = np.array(start), np.array(end) # just in case
    axes = ("x", "y", "z")
    # First try projecting to x boundary plane, then y boundary plane, then z boundary plane.
    for i, ax in enumerate(axes):
        planes = DETECTOR_BOUNDS[ax]
        end_target_plane = planes[0] if end[i] < start[i] else planes[1]
        tentative_proj_end = end + ((end_target_plane - end[i])/(end[i] - start[i]))*(end - start)

        # if the projection onto the x(y,z) boundary plane falls outside the y(z,x) or z(x,y) boundaries, try projecting onto another boundary. One of them has to work. 
        bounded1 = DETECTOR_BOUNDS[axes[i-1]][0] <= tentative_proj_end[i-1] <= DETECTOR_BOUNDS[axes[i-1]][1] 
        bounded2 = DETECTOR_BOUNDS[axes[i-2]][0] <= tentative_proj_end[i-2] <= DETECTOR_BOUNDS[axes[i-2]][1]
        if (not bounded1) or (not bounded2):
            continue
        else:
            proj_end = tentative_proj_end
    if not flipped:
        proj_start = project_segment_to_detector_bounds(end, start, flipped=True)
        return proj_start, proj_end
    else:
        return proj_end