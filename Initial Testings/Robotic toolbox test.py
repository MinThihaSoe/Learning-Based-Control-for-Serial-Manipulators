# pip install roboticstoolbox-python

import roboticstoolbox as rtb
import swift

#Initialize model

panda = rtb.models.Panda()

panda.plot(panda.qr, block=True)