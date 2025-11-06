# 1st 
# monkeypatch the context object with dummy values for a ctx object for the request
# ctx.constants.str
# ctx.parameters.date.format['mm-dd-yy'].no_quote  | cntx.parameters.date.yyyy-mm-dd 
# and so on for all python base datatypes except strings
# cntx.parameters.str.whitelist
# 2nd
# Write the python function to apply the context of the server and route constants to a template
# support date offsets and different date formats using a turing complete template
# consider if there is an appropiate templating library
# use python 3.13 or 3.12 compatable syntax

# ```sql
# SELECT * FROM '{{ctx.constants.str.source_path}}/.csv' # replace with whatever templating syntax we choose
# ```