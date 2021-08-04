import grunner as grun
import pprint

pp = pprint.PrettyPrinter()

# Print out the defaults
pp.pprint(grun.data) 

# Modify the defaults
grun.tag="grunexample"
grun.data["finaltime"] = 10.
grun.data["restart"] = "true"

# Print out the modifications 
pp.pprint(grun.data)

Ns = [32]
for N in Ns:
    grun.data["NX"] = N 
    grun.data["outputfiletag"]=grun.getdefault_filename()
    grun.run(dry_run=False, time=(N/16.)**3*0.05)
