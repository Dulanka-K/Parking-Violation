class DirectionIdentifier:

    def __init__(self):
        self.up = []
        self.down = []

    def getDirections(self, track):
        if track[-1][1] < track[0][1]:
            return 'up'
        elif track[-1][1] > track[0][1]:
            return 'down'
        else:
            return ''
