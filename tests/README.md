# inside-schools tests

## Working:
* fit_transmats.py: will fit and save the EI.obj and IR.obj transition matrix objects.  EI is the exposed to infected duration.  IR is the infected to recovered duration.
* is_covasim_seir.py: Run Covasim and SEIR side by side.  Results appear sensitive to EI.obj and IR.obj.
* controlled_schools.py: Run a controlled school.  Crazy slow.  Shows a plot on screen.

## Not working:
* controlled_covasim_seir.py: Run SEIR and Covasim side-by-side.  Currently out of date.
* test_controller.py: - Out of date script to run a simple school sim with the controller, probably should delete.
