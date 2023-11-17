#import nbcnews
import aljazeeranews

cawlers = [
    #nbcnews.nbcnews()
    aljazeeranews.aljazeeranews()
]

for cawler in cawlers:
    cawler.start()
