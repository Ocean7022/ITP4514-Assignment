import nbcnews, theStandard, aljazeeranews, cnnnews

cawlers = [
    #nbcnews.nbcnews(useJSON = False, onlyGetLinks = False),
    #theStandard.theStandard(useJSON = True, onlyGetLinks = False),
    #aljazeeranews.aljazeeranews(useJSON = True, onlyGetLinks = False),
    cnnnews.cnnnews(useJSON = False, onlyGetLinks = True)
]

for cawler in cawlers:
    cawler.start()
