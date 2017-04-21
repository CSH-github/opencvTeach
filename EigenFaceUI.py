#coding=utf-8
#sudo apt-get install python-wxgtk3.0
import wx
import  cStringIO
import EigenFaceText
class MyFrame(wx.Frame):  
  
    def __init__(self):  
        wx.Frame.__init__(self, parent=None,title="My Test Frame",pos = (100,100), size=(900,680))
          
       #添加第1个Panel面板

        self.panel1 = wx.Panel(parent=self,pos = (30,30), size=(120, 255))

        #添加第2个Panel面板  
        self.panel2 = wx.Panel(parent=self,pos = (185,30), size=(120*5+30, 130*4+100))
        #添加标签
        wx.StaticText(parent=self.panel2,label= "unit:e+15", pos=(10, 10),size=(150, 15))
        wx.StaticText(parent=self.panel1,label= "current picture", pos=(10, 10),size=(150, 15))

        #添加按钮
        self.btn=wx.Button(parent=self.panel1,label= "next",pos=(10, 200),size=(50, 25))
        self.btn.Bind(wx.EVT_BUTTON,  self.OnMyNextClick)

        self.btn2=wx.Button(parent=self,label= "star",pos=(150, 5),size=(150, 20))
        self.btn2.Bind(wx.EVT_BUTTON,  self.OnMyTestClick)
        self.btn.Enable(False)
        self.Centre() #居中显示  
        self.Show(True)#总是一创建就显示Frame框架,
        self.loadcount=0
        self.namecount=0

          
    def OnMyTestClick(self,event): #在按钮上面单击调用
        self.loadlist,self.namelist,self.dislist=EigenFaceText.TextEigenFace()
        wx.StaticText(parent=self,label= 'ok,please click next', pos=(320, 5),size=(250, 20))
        self.btn.Enable(True)


    def OnQuit(self, event): #点击退出菜单时调用  
        self.Close()

    def LoadLeftImage(self,filename):
        #添加图片
        imageFile = filename
        data = open(imageFile, "rb").read()
        stream = cStringIO.StringIO(data)
        bmp = wx.BitmapFromImage( wx.ImageFromStream( stream ))
        wx.StaticBitmap(self.panel1, -1, bmp, (10, 50))

    def LoadRightImage(self,filename):
        #添加图片
        imageFile = filename
        data = open(imageFile, "rb").read()
        stream = cStringIO.StringIO(data)
        bmp = wx.BitmapFromImage( wx.ImageFromStream( stream ))
        wx.StaticBitmap(self.panel2, -1, bmp, (10, 30))

    def OnMyNextClick(self,evevt):
        self.LoadLeftImage(self.loadlist[self.loadcount])
        length = len(self.namelist)
        j=0
        for i in range(20*self.namecount,20*self.namecount+20):
        #添加图片
           imageFile = self.namelist[i]
           data = open(imageFile, "rb").read()
           stream = cStringIO.StringIO(data)
           bmp = wx.BitmapFromImage( wx.ImageFromStream( stream ))
           wx.StaticBitmap(self.panel2, -1, bmp, (10+120*(j%5), 40+140*int(j/5)))
           wx.StaticText(parent=self.panel2,label= str(j+1), pos=(10+120*(j%5), 40+140*int(j/5)),size=(15, 15))
           wx.StaticText(parent=self.panel2,label= str(self.dislist[j]), pos=(10+120*(j%5), 160+140*int(j/5)),size=(120, 20))
           j=j+1
        self.loadcount=self.loadcount+1
        self.namecount=self.namecount+1

#################################################################################  
if __name__ == '__main__':
    app = wx.App()  
    frame = MyFrame()
    app.MainLoop()
