import nbcnews, theStandard, aljazeeranews

cawlers = [
    nbcnews.nbcnews(useJSON = False, onlyGetLinks = False),
    theStandard.theStandard(useJSON = False, onlyGetLinks = False),
    aljazeeranews.aljazeeranews(useJSON = False, onlyGetLinks = False)
]

for cawler in cawlers:
    cawler.start()
