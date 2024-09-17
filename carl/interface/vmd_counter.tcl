# TCL script to extract Z coordinates of res1 over all frames
set outfile [open "z_coords.txt" w]

# Loop over all frames
set num_frames [molinfo top get numframes]
for {set frame 0} {$frame < $num_frames} {incr frame} {
    # Go to the frame
    animate goto $frame
    
    # Get the Z coordinate of res1 (assuming res1 is identified by residue id 1)
    set sel [atomselect top "resid BRO1"]
    set z_coords [$sel get z]
    
    # Assuming res1 has only one atom, otherwise handle accordingly
    set z_coord [lindex $z_coords 0]
    
    # Write frame number and Z coordinate to file
    puts $outfile "$frame $z_coord"
    
    # Delete the selection to avoid memory issues
    $sel delete
}

close $outfile
