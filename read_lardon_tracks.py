import argparse
import os

import tables as tab
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps as mpl_colormaps
from glob import glob

## input, output configuration
# reco_files_dir, out_filename, out_plots_filename = sys.argv[1:]
parser = argparse.ArgumentParser(description="Reads .h5 files containing LARDON-reconstructed PDVD data, selects 3D tracks with PDS information and of certain topologies, and summarizes important info about these tracks in an output .h5 file.")
parser.add_argument("reco_files_dir", 
                    help="Directory containing all PDVD LARDON-reconstructed data .h5 files to read.")
parser.add_argument("--out-filename", 
                    default="out.h5",
                    help="Filename/filepath for output .h5 file.")
parser.add_argument("--plots-filename",  
                    default="out.pdf",
                    help="Filename/filepath for output plots.")
parser.add_argument('track_selection_condition',
                    nargs="?",
                    default="all",
                    choices=["all", "anode-crossers", "cathode-crossers", "anode-cathode-crossers", "anode-cathode-anode-crossers"],
                    help="Leave blank for all tracks. Selections are as inclusive as possible (i.e passing in 'anode-crossers' will return A, AC, CA and ACA tracks)")

args = parser.parse_args()

## Get input reco-data files
reco_filenames = sorted( glob( os.path.expandvars(args.reco_files_dir)+"/*.h5" ) )

## Create output h5 file 
out_file = tab.open_file(args.out_filename, mode="w")

### TRACK SELECTION CONDITIONS
## Stitching was found i.e. crosses cathode
stitched_tracks_condition = "((cathode_crosser_ID>-1))" 

## Stitching was found i.e. crosses cathode, and at least one half-track crosses its respective anode. 
ac_stitched_tracks_condition  = "("+"((is_anode_crosser==1)|(pairTrackIsAnodeCrosser==1))"+"&"+stitched_tracks_condition+")"

## Stitching was found i.e. crosses cathode, and both half-tracks cross their respective anode. 
aca_stitched_tracks_condition = "("+"((is_anode_crosser==1)&(pairTrackIsAnodeCrosser==1))"+"&"+stitched_tracks_condition+")"

## Crosses anode only (does not cross cathode)
a_crosser_only_condition = "( (is_anode_crosser==1) & (~(is_cathode_crosser==1)) )"

## The track was matched to a pds cluster
pds_cluster_match_condition  = "((cluster_ID>=0))"

### CHOOSING THE DESIRED TRACK TOPOLOGY SELECTION CONDITION
track_topo_selection_condition = ""
match args.track_selection_condition:
    case "all":
        track_topo_selection_condition="(ID>=0)"
    case "anode-crossers": # A, (AC, CA), ACA
        track_topo_selection_condition = "("+a_crosser_only_condition+"|"+ac_stitched_tracks_condition+")"
    case "cathode-crossers": # C, (AC, CA), ACA
        track_topo_selection_condition = stitched_tracks_condition
    case "anode-cathode-crossers": # (AC, CA), ACA
        track_topo_selection_condition = ac_stitched_tracks_condition
    case "anode-cathode-anode-crossers": # ACA only
        track_topo_selection_condition = aca_stitched_tracks_condition
    case _:
        raise argparse.ArgumentError
    
## For track counts 
track_topologies = [("ALL TRACKS"                                                         , "(ID>=0)"                    ),
                    ("Crosses anode, does not cross cathode"                              , a_crosser_only_condition     ),
                    ("Crosses cathode"                                                    , "(is_cathode_crosser==1)"    ), 
                    ("Crosses cathode and pair-track is found"                            , stitched_tracks_condition    ),
                    ("Crosses cathode + at least one anode, pair-track found (AC/CA/ACA)" , ac_stitched_tracks_condition ), 
                    ("Crosses cathode + both anodes, pair-track found (ACA)"              , aca_stitched_tracks_condition)]

### ITERATE THROUGH ALL INPUT FILES
for reco_filename in reco_filenames:
    data_file = tab.open_file(reco_filename, mode="r") # Read-only mode
    condensed_reco_filename = reco_filename.split(sep="/")[-1][:-3] # without directory and .h5

    tracks3d_table = data_file.root.tracks3d
    pds_peaks_table = data_file.root.pds_peaks
    pds_clusters_table = data_file.root.pds_clusters

    # print("tracks3d struct: ", tracks3d_table.dtype, "\n")
    # print("pds_clusters struct: ", pds_clusters_table.dtype, "\n")
    # print("pds_peaks struct: ", pds_peaks_table.dtype, "\n")

    # make sure the track ID works as an index of tracks3d table, for creation of pairTrackIsAnodeCrosser
    assert np.all( [tracks3d_table[i] == tracks3d_table.read_where('ID==i')[0] for i in range(tracks3d_table.nrows)] ) # even if only one row of the table matches the read_where condition, it will still return a structured np_array containing the row(s) matching the condition -- in this case an array of shape (1,): containing one element, one structured scalar corresponding to one row of the table. 

    # Array acting as an "extra column" of tracks3d_table 
    # For each track, it specifies whether or not its pair track in the other drift volume crosses that volume's anode. If this track has no pair track, it is -1. 
    pairTrackIsAnodeCrosser = np.array( [tracks3d_table.col('is_anode_crosser')[pairTrackID] 
                                         if pairTrackID!=-1 else -1 
                                         for pairTrackID in tracks3d_table.col('cathode_crosser_ID')] )


    ### TESTS AND STATISTICS
    for cRow in pds_clusters_table:
        # Make sure cathode_crosser_ID pairs and pds_cluster match_trk3D pairs are the same i.e. that individual clusters match to individual __physical__ tracks  
        mTrk3d = cRow['match_trk3D']
        if -1 not in mTrk3d:
            assert (tracks3d_table[ mTrk3d[0] ]['cathode_crosser_ID'] == mTrk3d[1]) and (tracks3d_table[ mTrk3d[1] ]['cathode_crosser_ID'] == mTrk3d[0])
        else:
            # if there is only one trackID in a cluster's match_trk3D, it is because that track has no cathode_crosser_ID (either because it actually doesn't cross cathode or the pairTrack could not be identified.)
            assert np.all( [(tracks3d_table[ti]['cathode_crosser_ID'] == -1) if ti>=0 else True for ti in mTrk3d] )

        # Make sure there is only one peak per channel per cluster (takes some time, can comment out) 
        cID = cRow['ID']
        peaks_from_cluster = pds_peaks_table.read_where('cluster_ID==cID')
        peak_counts_per_channel = np.unique(peaks_from_cluster['channel'], return_counts=True)[1]
        assert (peak_counts_per_channel == 1).all()

        # Make sure each cluster is matched to at most two tracks in tracks3d
        matchedTrackIDs = tracks3d_table.read_where('cluster_ID==cID')['ID']
        assert matchedTrackIDs.shape[0] <= 2
        if matchedTrackIDs.shape[0] > 0:
            assert np.all( [tID in mTrk3d for tID in matchedTrackIDs]  )

        # make sure the size column of pds_clusters means no. of peaks in cluster
        assert peak_counts_per_channel.sum() == cRow['size']

    # ensure that (is_cathode_crosser==1) is redundant with cathode_crosser_ID > -1
    assert tracks3d_table.read_where("cathode_crosser_ID>-1")["is_cathode_crosser"].all() 

    if "/track_counts" not in out_file:
        trk_ct_pair_dtype = np.dtype( [('all', 'i'), ('pds_cluster_matched', 'i')] )
        trk_cts_dtype = np.dtype( [('filename', 'U32'), 
                                   ('all_tracks_ct',            trk_ct_pair_dtype),   
                                   ('a_crossers_ct',            trk_ct_pair_dtype),
                                   ('c_crossers_ct',            trk_ct_pair_dtype),
                                   ('stitched_c_crossers_ct',   trk_ct_pair_dtype),
                                   ('stitched_ac_crossers_ct',  trk_ct_pair_dtype),
                                   ('stitched_aca_crossers_ct', trk_ct_pair_dtype)] )
        ## Here is where the table is actually created
        trk_ct_table = out_file.create_table(out_file.root, 'track_counts', trk_cts_dtype, "Track Counts (test)")
        # print(trk_ct_table.coldescrs)

    print(f"TRACKS3D STATISTICS FOR FILE {condensed_reco_filename}")
    trk_ct_table.row['filename'] = condensed_reco_filename
    for i, (txt, cond) in enumerate(track_topologies):
        n_tracks_selected       = tracks3d_table.read_where(cond).shape[0]
        n_tracks_selected_w_pds = tracks3d_table.read_where(cond+"&"+pds_cluster_match_condition).shape[0]
        # trk_ct_table.dtype.names[1:] are the six fields which actually store the track-counts for diff. conditions --> the filename field is excluded
        trk_ct_table.row[ trk_ct_table.dtype.names[1:][i] ] = (n_tracks_selected, n_tracks_selected_w_pds)
        
        print(f"{txt:<90}: {n_tracks_selected:<7} ---> with pds: {n_tracks_selected_w_pds:<7}")
    print("Note that for stitched tracks the number of actual physical tracks is the number of pairs i.e. divide by 2\n")
    trk_ct_table.row.append()
    trk_ct_table.flush()        # hopefully this is not wasteful. It will mean less memory usage by the program, but it may be slower as a result?????


    ### PERFORM TRACK SELECTION 
    slctd_tracks_w_pds_np = tracks3d_table.read_where(track_topo_selection_condition + "&" + pds_cluster_match_condition)

    # dtype consistency test
    assert tracks3d_table.dtype == slctd_tracks_w_pds_np.dtype
        # this is because of tables.Leaf.dtype. Also an equivalent way: tracks3d_table.dtype == tracks3d_table.description._v_dtype
    # sanity check for z correction
    assert (slctd_tracks_w_pds_np['z0_light'] != 9999).all()


    ### CREATE OUTPUT TABLE IF NOT EXISTENT
    if "/output" not in out_file:
        get_tracks3d_field_formats = lambda field_name_list : [ tracks3d_table.coldescrs[fname] for fname in field_name_list ]

        out_field_names   = ["filename"]
        out_field_formats = [tab.StringCol(itemsize=32)]
        
        tracks3d_fields_to_get_once = ['event', 'trigger', 'cluster_ID', 'ID', 'cathode_crosser_ID'] 
        out_field_names   += tracks3d_fields_to_get_once
        out_field_formats += get_tracks3d_field_formats(tracks3d_fields_to_get_once)
                                                # the cluster size thing is really for debug, it can be taken out easily
        out_field_names   += ['total_npe',      'cluster_size']
        out_field_formats += [tab.Float64Col(), tab.UInt32Col()]

        tracks3d_fields_to_get_twice = ['is_anode_crosser', 'is_cathode_crosser', 'x_ini', 'x_end', 'y_ini', 'y_end', 'z0_light', 'z_ini', 'z_end', 'theta_ini', 'theta_end', 'phi_ini', 'phi_end'] 
        tracks3d_fields_to_get_twice_formats = get_tracks3d_field_formats(tracks3d_fields_to_get_twice)
        tracks3d_fields_to_get_twice         += ['z_ini_corrected', 'z_end_corrected']
        tracks3d_fields_to_get_twice_formats += [tab.Float32Col(),  tab.Float32Col()]

        out_field_names   += tracks3d_fields_to_get_twice + ["cc_"+fname for fname in tracks3d_fields_to_get_twice] 
        out_field_formats += tracks3d_fields_to_get_twice_formats*2

        # make sure the columns are ordered in the output table consistent with how we ordered them here
        for i, col in enumerate(out_field_formats):
            col._v_pos = i

        out_descr = { field_name: field_format for field_name, field_format in zip(out_field_names, out_field_formats) }
        ## Here is where the table is actually created
        out_table = out_file.create_table(out_file.root, 'output', out_descr, "Output (test)")
        out_table.attrs.selection_condition = args.track_selection_condition

        assert out_table.dtype['z_ini_corrected'] == slctd_tracks_w_pds_np.dtype['z_ini'] 
        assert out_table.dtype['z_end_corrected'] == slctd_tracks_w_pds_np.dtype['z_end'] 
        assert out_table.dtype['total_npe'] ==       pds_peaks_table.dtype['npe'] 

        print("Output dtype:", out_table.dtype, "\n")

    prev_out_nrows = out_table.nrows    ## FOR DEBUG
    ### FILL IN OUTPUT TABLE
    next_row = out_table.row
    visited_tracks = np.zeros_like(slctd_tracks_w_pds_np, np.bool_)
    for i, track in enumerate(slctd_tracks_w_pds_np):   # track is a structured scalar of the structured np dtype of slctd_tracks_w_pds_np, which is the same as the structured np dtype corresponding to the Description / struct of the table tracks3d.
        if not visited_tracks[i]:
            trackID, pairTrackID, clsID, trig = track['ID'], track['cathode_crosser_ID'], track['cluster_ID'], track['trigger'] 
            
            # Get pair (stitched) track
            if pairTrackID != -1:
                pairTrackIndex = np.argwhere(slctd_tracks_w_pds_np['ID'] == pairTrackID)[0] # remember arr[arr==x] --> array([x]), not x itself, hence why we need the [0]
                pairTrack = slctd_tracks_w_pds_np[pairTrackIndex]       
                # Sanity checks, can comment out atfter first execution
                # making sure cathode_crosser_ID pairs and pds_cluster match_trk3D pairs are the same.
                this_cluster_match_trk3d_list = pds_clusters_table.read_where("ID == clsID")[0]["match_trk3D"]
                assert (trackID in this_cluster_match_trk3d_list) and (pairTrackID in this_cluster_match_trk3d_list)
                assert np.all([ track[fname]==pairTrack[fname] for fname in ['event', 'trigger', 'cluster_ID'] ])
                assert pairTrack['cathode_crosser_ID'] == trackID
                ## The "pairTrack" (cc_) is in the upper volume while the "regular" track is in the lower volume:
                assert pairTrack['t_ini'] < pairTrack['t_end']  
                assert track['t_ini']     > track['t_end']  

            # fill in known values, for both track and pairTrack 
            for colname in out_table.colnames:              # iterate over output table's fields
                if (colname[:3] == 'cc_'):
                    if (pairTrackID == -1):
                        continue
                    elif colname[3:] in track.dtype.names:  # is this field (without the cc_) already among the fields of the (down-selected) tracks3d table?
                        next_row[colname] = pairTrack[ colname[3:] ]
                elif colname in track.dtype.names:          # is this field already among the fields of the (down-selected) tracks3d table?            
                    next_row[colname] = track[colname]                  
            
            ### FIND LIGHT-CORRECTED START AND END Z FOR BOTH TRACK AND PAIRTRACK
            next_row['z_ini_corrected'] = track['z_ini'] + track['z0_light']
            next_row['z_end_corrected'] = track['z_end'] + track['z0_light']
            if pairTrackID != -1:
                next_row['cc_z_ini_corrected'] = pairTrack['z_ini'] + pairTrack['z0_light']
                next_row['cc_z_end_corrected'] = pairTrack['z_end'] + pairTrack['z0_light']

                # insanity check
                assert (pairTrack['z_ini']==next_row['cc_z_ini']) and (pairTrack['z_end']==next_row['cc_z_end']) and (pairTrack['z0_light']==next_row['cc_z0_light']) 

            ### FIND TOTAL nPE ASSOCIATED WITH THIS TRACK        
            pds_peaks_this_cluster_np = pds_peaks_table.read_where("(cluster_ID==clsID)&(trigger==trig)")
            total_nPE = pds_peaks_this_cluster_np['npe'].sum()
            next_row['total_npe'] = total_nPE

            # the cluster size thing is really for debug, it can be taken out easily
            this_cluster_size = pds_clusters_table.read_where("(ID==clsID)&(trigger==trig)")[0]["size"]
            assert this_cluster_size == len(pds_peaks_this_cluster_np)
            next_row['cluster_size'] = this_cluster_size

            next_row['filename'] = condensed_reco_filename
            next_row.append()

            visited_tracks[i] = True
            if pairTrackID != -1:
                visited_tracks[pairTrackIndex] = True        

    out_table.flush()

    # sanity check
    if len(slctd_tracks_w_pds_np) > 0:
        assert out_table.col('filename')[-1] == condensed_reco_filename.encode(encoding='ascii') # strings in PyTables tables are stored as strings of byte literals, not of python unicode literals. Hence why we need to find the bytestring for condensed_reco_filename using encode().
    
    print("Number of rows selected from tracks3d: ", len(slctd_tracks_w_pds_np))
    n_matched_clusters = len(np.unique(slctd_tracks_w_pds_np['cluster_ID']))
    print("Number of pds_clusters matched to rows selected from tracks3d: ", n_matched_clusters)
    #######################################
    if args.track_selection_condition=='anode-cathode-crossers':
        assert len(slctd_tracks_w_pds_np) == trk_ct_table.col("stitched_ac_crossers_ct")['pds_cluster_matched'][-1]
    elif args.track_selection_condition=='anode-cathode-anode-crossers':
        assert len(slctd_tracks_w_pds_np) == trk_ct_table.col("stitched_aca_crossers_ct")['pds_cluster_matched'][-1]
    elif args.track_selection_condition=='anode-crossers':
        assert len(slctd_tracks_w_pds_np) == trk_ct_table.col("stitched_ac_crossers_ct")['pds_cluster_matched'][-1] + trk_ct_table.col("a_crossers_ct")['pds_cluster_matched'][-1] 

    if args.track_selection_condition not in ('all', 'anode-crossers'): # anode-only crossers have no pair track
        assert n_matched_clusters == len(slctd_tracks_w_pds_np) / 2
    #######################################
    print("Number of rows in out_table: ", out_table.nrows)
    assert out_table.nrows == prev_out_nrows + n_matched_clusters
    print()

    data_file.close()

### PRINT OVERALL TRACK STATISTICS 
print(f"OVERALL TRACKS3D STATISTICS FOR ALL READ FILES")
for i, (txt, _) in enumerate(track_topologies):
    this_slctn_cts = trk_ct_table.col( trk_ct_table.dtype.names[1:][i] )  # remember these are the fields of trk_ct_tabl which actually store the counts for the six conditions: the first field just stores the filename.
    n_tracks_selected       = this_slctn_cts['all'].sum() # table.col("...") returns an np array of the column, which will be a structured array in the case of nested columns, of struct-dtype given by that nested column's Description. 
    n_tracks_selected_w_pds = this_slctn_cts['pds_cluster_matched'].sum() # remember: each row of trk_ct_table -- and subsequently each entry of its column this_slctn_cts -- corresponds to one input file, we sum over the column to get the total across all files. 
    print(f"{txt:<90}: {n_tracks_selected:<7} ---> with pds: {n_tracks_selected_w_pds:<7}")
print("Note that for stitched tracks the number of actual physical tracks is the number of pairs i.e. divide by 2\n")

### PRINT OUT SOME COLUMNS FROM out_table, MOST IMPORTANTLY THE ClusterID and Total nPE.
print("Topology selection condition used: ", out_table.attrs.selection_condition,"\n")
print("View of Output Table")
print(f"{'':<7}{'File':<35}{'Trigger':<15}{'Track ID':<15}{'PairTrack ID':<15}{'Cluster ID':<15}{'Total nPE':<20}")
for i, out_row in enumerate(out_table):
    # we have to decode the byte-string back to a string of unicode characters
    print(f"{i:<7}{out_row['filename'].decode('utf8'):<35}{out_row['trigger']:<15}{out_row['ID']:<15}{out_row['cathode_crosser_ID']:<15}{out_row['cluster_ID']:<15}{out_row['total_npe']:<20}")
print()

### IGNORE TRACKS WHICH HAVE START/END POINTS WITH RECONSTRUCTED XYZ COORDS OUTSIDE REASONABLE RANGE
min_z_bound, max_z_bound = (-345, 345) # this is actually more strict than the LArSoft detector bound. 
min_x_bound, max_x_bound = (-425, 425) #|--> these two are the detector bounds as per LArSoft
min_y_bound, max_y_bound = (-300, 600) #|
oobz_condition = '|'.join( [ '|'.join( [f"({z_coord}{z_cond})" for z_coord in ("z_ini_corrected", "z_end_corrected", "cc_z_ini_corrected", "cc_z_end_corrected")] ) for z_cond in (" > max_z_bound", " < min_z_bound") ] )  
oobx_condition = '|'.join( [ '|'.join( [f"({x_coord}{x_cond})" for x_coord in ("x_ini", "x_end", "cc_x_ini", "cc_x_end")] ) for x_cond in (" > max_x_bound", " < min_x_bound") ] )  
ooby_condition = '|'.join( [ '|'.join( [f"({y_coord}{y_cond})" for y_coord in ("y_ini", "y_end", "cc_y_ini", "cc_y_end")] ) for y_cond in (" > max_y_bound", " < min_y_bound") ] )  
out_table_np = out_table.read_where("~(" + oobz_condition + '|' + oobx_condition + '|' + ooby_condition + ")")
print(f"Tracks (pairs) with start/end coords out of range [({min_x_bound},{max_x_bound}),({min_y_bound},{max_y_bound}),({min_z_bound},{max_z_bound})]: {out_table.nrows - out_table_np.shape[0]}")
print()

### PLOT SELECTED TRACKS
fig, axs = plt.subplots(2, figsize=(8, 16), layout="constrained")
axs[0].set_xlabel("x [cm]")
axs[0].set_ylabel("z [cm]")
axs[0].plot([-334, 334], [ 338.5,  338.5], ls="-",  lw=3, color="orange", label="Anode")
axs[0].plot([-334, 334], [-338.5, -338.5], ls="-",  lw=3, color="orange")
axs[0].plot([-334, 334], [0,       0    ], ls="--", lw=3, color="grey",   label="Cathode")

axs[1].set_xlabel("y [cm]")
axs[1].set_ylabel("z [cm]")
axs[1].plot([0, 300], [ 338.5,  338.5], ls="-",  lw=3, color="orange", label="Anode")
axs[1].plot([0, 300], [-338.5, -338.5], ls="-",  lw=3, color="orange")
axs[1].plot([0, 300], [0,       0    ], ls="--", lw=3, color="grey",   label="Cathode")

# max_total_npe = np.max( out_table.col('total_npe') )
# min_total_npe = np.min( out_table.col('total_npe') )
# for i, track in enumerate(out_table): # This time, track is a Row instance of out_table, not a structured scalar from slctd_tracks_w_pds_np  
max_total_npe = np.max( out_table_np['total_npe'] )
min_total_npe = np.min( out_table_np['total_npe'] )
tracks_lw = 0.8
tracks_alpha = 0.6
tracks_to_plot = out_table_np if len(out_table_np) < 50000 else out_table_np[::10] # Only plot every 10th track if there are too many
for i, track in enumerate(tracks_to_plot): # Now track is a structured scalar from out_table_np (tracks_to_plot), not a structured scalar from slctd_tracks_w_pds_np  
    ## Just color tracks based on their order in the table
    # trkColor = mpl_colormaps['viridis'](i / out_table.nrows)
    ## Color tracks proportional to their associated total npe   
    trkColor = mpl_colormaps['plasma_r']((track['total_npe'] - min_total_npe) / (max_total_npe - min_total_npe)) # The bluer the track, the more its npe
    trkColor = (*trkColor[:-1], tracks_alpha) # make them translucent 
    # print(trkColor)

    axs[0].plot([track['x_ini'],           track['x_end']], 
                [track['z_ini_corrected'], track['z_end_corrected']], ls=":", lw=tracks_lw, color=trkColor)
    axs[1].plot([track['y_ini'],           track['y_end']], 
                [track['z_ini_corrected'], track['z_end_corrected']], ls=":", lw=tracks_lw, color=trkColor)
    
    if track['cathode_crosser_ID'] >= 0:
        axs[0].plot([track['cc_x_ini'],           track['cc_x_end']], 
                    [track['cc_z_ini_corrected'], track['cc_z_end_corrected']], ls=":", lw=tracks_lw, color=trkColor)
        axs[1].plot([track['cc_y_ini'],           track['cc_y_end']], 
                    [track['cc_z_ini_corrected'], track['cc_z_end_corrected']], ls=":", lw=tracks_lw, color=trkColor)

wPairMask = (tracks_to_plot['cathode_crosser_ID'] >= 0)
axs[0].plot(tracks_to_plot['x_ini'],               tracks_to_plot['z_ini_corrected'],               "o", color="pink",   alpha=tracks_alpha, ms=2, label="Ini points")
axs[0].plot(tracks_to_plot['x_end'],               tracks_to_plot['z_end_corrected'],               "o", color="yellow", alpha=tracks_alpha, ms=2, label="End points")
axs[0].plot(tracks_to_plot['cc_x_ini'][wPairMask], tracks_to_plot['cc_z_ini_corrected'][wPairMask], "o", color="pink",   alpha=tracks_alpha, ms=2)
axs[0].plot(tracks_to_plot['cc_x_end'][wPairMask], tracks_to_plot['cc_z_end_corrected'][wPairMask], "o", color="yellow", alpha=tracks_alpha, ms=2)

axs[1].plot(tracks_to_plot['y_ini'],               tracks_to_plot['z_ini_corrected'],               "o", color="pink",   alpha=tracks_alpha, ms=2, label="Ini points")
axs[1].plot(tracks_to_plot['y_end'],               tracks_to_plot['z_end_corrected'],               "o", color="yellow", alpha=tracks_alpha, ms=2, label="End points")
axs[1].plot(tracks_to_plot['cc_y_ini'][wPairMask], tracks_to_plot['cc_z_ini_corrected'][wPairMask], "o", color="pink",   alpha=tracks_alpha, ms=2)
axs[1].plot(tracks_to_plot['cc_y_end'][wPairMask], tracks_to_plot['cc_z_end_corrected'][wPairMask], "o", color="yellow", alpha=tracks_alpha, ms=2)

# wPairMask = (out_table.col('cathode_crosser_ID') >= 0)
# axs[0].plot(out_table.col('x_ini'),               out_table.col('z_ini_corrected'),               "o", color="pink",   ms=3, label="Ini points")
# axs[0].plot(out_table.col('x_end'),               out_table.col('z_end_corrected'),               "o", color="yellow", ms=3, label="End points")
# axs[0].plot(out_table.col('cc_x_ini')[wPairMask], out_table.col('cc_z_ini_corrected')[wPairMask], "o", color="pink",   ms=3)
# axs[0].plot(out_table.col('cc_x_end')[wPairMask], out_table.col('cc_z_end_corrected')[wPairMask], "o", color="yellow", ms=3)

# axs[1].plot(out_table.col('y_ini'),               out_table.col('z_ini_corrected'),               "o", color="pink",   ms=3, label="Ini points")
# axs[1].plot(out_table.col('y_end'),               out_table.col('z_end_corrected'),               "o", color="yellow", ms=3, label="End points")
# axs[1].plot(out_table.col('cc_y_ini')[wPairMask], out_table.col('cc_z_ini_corrected')[wPairMask], "o", color="pink",   ms=3)
# axs[1].plot(out_table.col('cc_y_end')[wPairMask], out_table.col('cc_z_end_corrected')[wPairMask], "o", color="yellow", ms=3)

axs[0].legend()
axs[1].legend()
fig.savefig(args.plots_filename)
plt.close()

### PLOT HISTOGRAM OF SELECTED TRACK nPEs
plt.figure() 
plt.xlabel("Total nPE associated with selected track")
plt.ylabel("No. of selected tracks")
npe_hist = plt.hist( out_table_np['total_npe'], bins="auto", histtype="stepfilled" )
assert npe_hist[0].sum() == out_table_np.shape[0]
plt.plot([],[]," ", label=f"bin width = {npe_hist[1][1] - npe_hist[1][0]:.0f}") 
plt.legend()
plt.savefig(args.plots_filename.split(sep=".")[0] + "_npe_hist." + args.plots_filename.split(sep=".")[1])
plt.close()

### PLOT TRACK TOTAL nPE vs TRACK "LENGTH"
# track_lengths = np.maximum( (out_table.col('cc_x_ini') - out_table.col('x_end'))**2 
#                           + (out_table.col('cc_y_ini') - out_table.col('y_end'))**2 
#                           + (out_table.col('cc_z_ini_corrected') - out_table.col('z_end_corrected'))**2, 
#                             (out_table.col('x_ini') - out_table.col('cc_x_end'))**2 
#                           + (out_table.col('y_ini') - out_table.col('cc_y_end'))**2 
#                           + (out_table.col('z_ini_corrected') - out_table.col('cc_z_end_corrected'))**2 
#                           )**(1/2)

def calc_track_length(trk):
    is_pair = trk['cathode_crosser_ID'] != -1
    return ((trk['cc_x_ini'           if is_pair else 'x_ini']           - trk['x_end'])**2 
          + (trk['cc_y_ini'           if is_pair else 'y_ini']           - trk['y_end'])**2 
          + (trk['cc_z_ini_corrected' if is_pair else 'z_ini_corrected'] - trk['z_end_corrected'])**2 )**(1/2) 

track_lengths = np.array( [calc_track_length(trk) for trk in out_table_np] )

plt.figure()
plt.xlabel("Total cathode-crossing track 'length' (start-end distance)")
plt.ylabel("Total nPE associated with whole cathode-crossing track")
plt.plot(track_lengths, out_table_np['total_npe'], ".", ms=1, alpha=0.8)
plt.savefig(args.plots_filename.split(sep=".")[0] + "_npe_vs_trkLengths." + args.plots_filename.split(sep=".")[1])
plt.close()

plt.figure()
plt.xlabel("Number of peaks in pds cluster matched to track")
plt.ylabel("Total nPE associated with whole cathode-crossing track")
plt.plot(out_table_np['cluster_size'], out_table_np['total_npe'], ".", ms=1, alpha=0.8)
plt.savefig(args.plots_filename.split(sep=".")[0] + "_npe_vs_clusterSize." + args.plots_filename.split(sep=".")[1])
plt.close()

## ANGULAR INFORMATION
start_phis = np.array([trk['phi_ini'] if trk['cathode_crosser_ID'] == -1 else trk['cc_phi_ini'] for trk in out_table_np])
end_phis = out_table_np['phi_end']
assert np.all((180>=start_phis)&(start_phis>=-180)) and np.all((180>=end_phis)&(end_phis>=-180))

start_thetas = np.array([trk['theta_ini'] if trk['cathode_crosser_ID'] == -1 else trk['cc_theta_ini'] for trk in out_table_np])
end_thetas = out_table_np['theta_end']
assert np.all((180>=start_thetas)&(start_thetas>=-180)) and np.all((180>=end_thetas)&(end_thetas>=-180))

phi_bw = 10 # degrees
phi_bin_edges = np.linspace(-180, 180, (360 // phi_bw)+1)
theta_bw = 5 # degrees
theta_bin_edges = np.linspace(90, 180, (90 // theta_bw)+1)

plt.figure()
_,_,_,im = plt.hist2d(start_phis, 
           start_thetas, 
           bins=[phi_bin_edges, theta_bin_edges])
plt.colorbar(im)
plt.xlabel(r"Start $\varphi$ [degrees]")
plt.ylabel(r"Start $\theta$ [degrees]")
plt.savefig(args.plots_filename.split(sep=".")[0] + "_start_phi_theta_hist." + args.plots_filename.split(sep=".")[1])
plt.close()

plt.figure()
_,_,_,im = plt.hist2d(end_phis, 
           end_thetas, 
           bins=[phi_bin_edges, theta_bin_edges])
plt.colorbar(im)
plt.xlabel(r"End $\varphi$ [degrees]")
plt.ylabel(r"End $\theta$ [degrees]")
plt.savefig(args.plots_filename.split(sep=".")[0] + "_end_phi_theta_hist." + args.plots_filename.split(sep=".")[1])
plt.close()


out_file.close()