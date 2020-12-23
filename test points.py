
# -*- coding: utf-8 -*-

from microMLP   import MicroMLP
from tkinter    import *
from threading  import *

# ----------------------------------------------------------------

width           = 800   # Window/Canvas width
height          = 500   # Window/Canvas height
examples        = [ ]

# ----------------------------------------------------------------

def rgb2hex(rgb):
    return '#%02x%02x%02x' % rgb 

# ----------------------------------------------------------------

def onCanvasClick(evt) :
    examples.append((evt.x, evt.y))
    can.create_oval( evt.x-7, evt.y-7, evt.x+7, evt.y+7,
                     fill    = '#3366AA',
                     outline = '#AA3366',
                     width   = 2 )

# ----------------------------------------------------------------

class processThread(Thread) :

    def run(self) :
        evt  = Event()
        line = None
        while not evt.wait(0.010) :
            if len(examples) >= 2 :
                for i in range(30) :
                    for ex in examples :
                        mlp.Learn( [ MicroMLP.NNValue.FromAnalogSignal(ex[0]/width) ],
                                   [ MicroMLP.NNValue.FromAnalogSignal(ex[1]/height) ] )
                pts = [ ]
                for x in range(0, width, 10) :
                    out = mlp.Predict([MicroMLP.NNValue.FromAnalogSignal(x/width)])
                    y   = out[0].AsFloat * height
                    pts.append((x, y))
                can.delete(line)
                line = can.create_line(pts, fill='#3366AA')

# ----------------------------------------------------------------

mlp = MicroMLP.Create( neuronsByLayers           = [1, 15, 1],
                       activationFuncName        = MicroMLP.ACTFUNC_TANH,
                       layersAutoConnectFunction = MicroMLP.LayersFullConnect )

mainWindow = Tk()
mainWindow.title('microMLP - test points')
mainWindow.geometry('%sx%s' % (width, height))
mainWindow.resizable(False, False)

can = Canvas( mainWindow,
              width       = width,
              height      = height,
              bg          = 'white',
              borderwidth = 0 )
can.bind('<Button-1>', onCanvasClick)
can.pack()

pc = processThread()
pc.daemon = True
pc.start()

mainWindow.mainloop()
