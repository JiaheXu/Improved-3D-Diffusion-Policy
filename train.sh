ns_arr=("handover_block" "open_marker" "stack_blocks" "ziploc" "lift_ball" "straighten_rope" "pickup_plate" "stack_bowls" "insert_marker_into_cup" "insert_battery")
# "insert_battery"
#ns_arr=( "stack_blocks") #"stack_blocks" "lift_ball" "pickup_plate" "stack_bowls" "straighten_rope")
for ns in "${ns_arr[@]}"
do
    #python3 data_processing_bimanual_idp3.py -t $ns -d 20
    bash scripts/3dda_data.sh idp3 idp3_bimanual $ns
done

# hard

# handover_block
# close_marker
# stack_blocks
# ziploc #need to customize EE
# insert_battery

# easy

# lift_ball #need to customize EE
# rope
# pickup_plate # might need to customize EE
# stack_bowls
# insert_marker_into_cup
