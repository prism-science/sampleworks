"""Find structural contexts like each altloc conformer of a loop with the FoldDisco webserver.

Modules
-------
loop_motif  split a residue set's two altlocs and choose a distinguishing motif
query       FoldDisco query files and the search.foldseek.com API
summary     count close matches per database and conformer
pipeline    the whole run, shared by the web page and the command line
server      local web interface
cli         ``folddisco-altloc`` command
"""
