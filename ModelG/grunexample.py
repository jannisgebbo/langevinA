import grunner as grun
import pprint

pp = pprint.PrettyPrinter()

# Print out the defaults
pp.pprint(grun.data) 

# Modify the defaults
grun.tag="grunexample"
grun.data["finaltime"] = 10.

# Print out the modifications 
pp.pprint(grun.data)

grun.data["NX"] =32
grun.data["outputfiletag"]=grun.getdefault_filename()
grun.run(dry_run=True, ncpus="1")
