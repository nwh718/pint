from pint import UnitRegistry

ureg = UnitRegistry()

# test with an alias
unit1 = ureg.get_unit_by_alias('kg')
print("kg ->", unit1)

# test with a unit name itself
unit2 = ureg.get_unit_by_alias('kilogram')
print("kilogram ->", unit2)

# test with unknown alias
unit3 = ureg.get_unit_by_alias('unknown_alias')
print("unknown_alias ->", unit3)
