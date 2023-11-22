import nbcnews, theStandard, aljazeeranews

cawlers = [
    #nbcnews.nbcnews(useJSON = False, onlyGetLinks = False),
    #theStandard.theStandard(useJSON = True, onlyGetLinks = False),
    aljazeeranews.aljazeeranews(useJSON = True, onlyGetLinks = False)
]

for cawler in cawlers:
    cawler.start()
