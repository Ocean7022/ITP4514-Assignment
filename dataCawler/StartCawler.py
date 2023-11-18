import nbcnews, theStandard, aljazeeranews

cawlers = [
    #nbcnews.nbcnews(useJSON = False, onlyGetLinks = True),
    #theStandard.theStandard(useJSON = False, onlyGetLinks = True),
    aljazeeranews.aljazeeranews(useJSON = False, onlyGetLinks = True)
]

cawlers[0].start()

#for cawler in cawlers:
#    cawler.start()
