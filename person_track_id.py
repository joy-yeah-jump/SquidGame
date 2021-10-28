class PersonTrackId :
    # id = -1
    # ix1, iy1, ix2, iy2 = -1      # first coordinate
    # lx1, ly1, lx2, ly2 = -1      # last coordinate
    # inx1, iny1, inx2, iny2 = -1  # intersection point
    # now = -1                     # now(last) area
    # inter = -1                   # intersection area
    #
    # def __init__(self, id, ix1, iy1, ix2, iy2):
    #     self.id = id
    #     self.ix1 = ix1
    #     self.iy1 = iy1
    #     self.ix2 = ix2
    #     self.iy2 = iy2
    #
    # def set_coord(self, id, lx1, ly1, lx2, ly2):
    #     if id != self.id :
    #         print("id not same")
    #         pass
    #     self.lx1 = lx1
    #     self.ly1 = ly1
    #     self.lx2 = lx2
    #     self.ly2 = ly2
    #
    #     self.inx1 = max(self.ix1, self.lx1)
    #     self.iny1 = max(self.iy1, self.ly1)
    #     self.inx2 = min(self.ix2, self.lx2)
    #     self.iny2 = min(self.iy2, self.ly2)
    #
    #     if self.inx1 < self.inx2 and self.iny1 < self.iny2 :
    #         self.inter = (self.inx2 - self.inx1) * (self.iny2 - self.iny1)
    #     else : self.inter = -1
    pass