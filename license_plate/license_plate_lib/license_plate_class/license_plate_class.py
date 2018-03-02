

class ANPRChar:
    def __init__(self, contour, location, prediction, digitprediction, parent_index):
        self.contour = contour
        self.location = location
        self.x = location[0]
        self.y = location[1]
        self.w = location[2]
        self.h = location[3]
        self.prediction = prediction
        self.digitprediction = digitprediction
        self.parent_index = parent_index
        # 각 contour들의 관계
        self.prevDiffX = 0
        self.prevDiffY = 0
        self.nextDiffX = 0
        self.nextDIffY = 0
        # temp variables
        self.group_y_count = 0
        # height minimum over count
        self.group_h_count = 0
