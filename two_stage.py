from PIL import Image
import RPi.GPIO as GPIO
import pigpio
import time
import os

from yolo import YOLO


class LevelTracker:
    """The meta component for tracking the level position."""

    UpBdSwid = 23
    LoBdSwid = 24
    StepPins = [4, 17, 27, 22]  # IN1~4
    Seq = [[1, 0, 0, 0],
           [1, 1, 0, 0],
           [0, 1, 0, 0],
           [0, 1, 1, 0],
           [0, 0, 1, 0],
           [0, 0, 1, 1],
           [0, 0, 0, 1],
           [1, 0, 0, 1]]

    StepCount = len(Seq)

    def __init__(self):
        self.total_move_distance = 0
        self.img_id = 1
        self.pixel_step_coeff = None
        self.img_size = [320, 240]
        self.yolo = YOLO()
        self._show_message("Detection model loaded.", 6)
        self.pi = pigpio.pi()

        if not os.path.exists(os.path.join('.', 'tmpshot')):
            os.mkdir(os.path.join('.', 'tmpshot'))

        self._show_message("Start tracker service", 6)
        self._hwsetup()
        self._hwinit()

    def firstStage(self):
        """First stage: move by fixed steps until near the level position.

        :return: None for normally exit. 300 for initialization.
        """

        self._show_message("Start first stage", 1)
        if self._move(1, 1.5, 8):
            self._show_message("Move 10 mm upward first", 1)
            self._show_message("Total distance = {} mm".format(
                                str(self.total_move_distance)), 6)

        while True:
            try:
                img = self._take_shot()
                while img is None:
                    time.sleep(1)
                    img = self._take_shot()
                self.img_id += 1
                _distance = self.recognize_level(img)
                if _distance is None:
                    if self._move(1, 1.5, 16):
                        self._show_message("Level not found, move 20 mm upward", 1)
                        self._show_message("Total distance = " + str(self.total_move_distance) + " mm", 6)
                        continue
                    else:
                        self._show_message("Reach 310 mm, restart all", 1)
                        self._hwinit()
                        return 300
                else:
                    self._show_message("First stage finished with distance = " + str(_distance) + " px", 1)
                    return _distance
            except Exception as e:
                self._show_message("Type: " + str(type(e)) + " | Msg: " + str(e), -1)
                continue

    def secondStage(self):
        """Second stage: keep track the level by two steps.

        :return: -1 for initialization
        """
        while True:
            try:
                img = self._take_shot()
                while img is None:
                    time.sleep(1)
                    img = self._take_shot()
                self.img_id += 1
                old_distance = self.recognize_level(img)

                if old_distance is None:
                    self._show_message("Level lost, restart all", 2)
                    self._hwinit()
                    return -1

                if -3 <= old_distance <= 3:
                    self._show_message("Already in central region (" + str(old_distance) + " px), no need to move", 2)
                    self._show_message("Total distance = " + str(self.total_move_distance) + " mm", 6)
                    continue
                if self.pixel_step_coeff is None:
                    # find conversion between pixel and mm
                    if old_distance > 0:
                        if self._move(1, 1.5, 4):
                            self._show_message("First step, move 5 mm upward", 2)
                            self._show_message("Total distance = " + str(self.total_move_distance) + " mm", 6)
                        else:
                            self._show_message("Reach 310 mm, restart all", 2)
                            self._hwinit()
                            return -1
                    elif old_distance < 0:
                        if self._move(-1, 1.5, 4):
                            self._show_message("First step, move 5 mm downward", 2)
                            self._show_message("Total distance = " + str(self.total_move_distance) + " mm", 6)

                    img = self._take_shot()
                    while img is None:
                        time.sleep(1)
                        img = self._take_shot()
                    self.img_id += 1
                    new_distance = self.recognize_level(img)
                    if new_distance is None:
                        self._show_message("Level lost, restart all", 2)
                        self._hwinit()
                        return -1
                    self.pixel_step_coeff = (old_distance - new_distance) / 4
                    if -3 <= new_distance <= 3:
                        self._show_message("Already in central region (" + str(new_distance) + " px), no need to move", 2)
                        self._show_message("Total distance = " + str(self.total_move_distance) + " mm", 6)
                        continue
                    step = self._px2step(new_distance)
                else:
                    step = self._px2step(old_distance)
                if self._move(1, 1.5, step):
                    self._show_message("Move " + str(step * 1.25) + " mm", 2)
                    self._show_message("Total distance = " + str(self.total_move_distance) + " mm", 6)
                else:
                    self._show_message("Reach 310 mm, restart all", 2)
                    self._hwinit()
                    return -1
            except Exception as e:
                self._show_message("Type: " + str(type(e)) + " | Msg: " + str(e), -1)
                continue

    def recognize_level(self, img):
        """Recognize the image and determine level position.

        :param img: Input image.
        :return: Distance from central. Positive for above, negative for below, none for not found.
        """

        boxes, scores = self.yolo.detect(img)
        if len(scores) < 1:
            return None
        target_position = self._define_target_position(boxes)
        return self.img_size[0] // 2 - target_position

    def _px2step(self, pixel):
        step = pixel / self.pixel_step_coeff
        if step - 0.2 * (step // 0.2) >= 0.1:
            return (step // 0.2 + 1) * 0.2
        else:
            return (step // 0.2) * 0.2

    def _define_target_position(self, boxes):
        """Function to determine the final prediction of level position

        :param boxes: boxes with their position information
        :return the final level position
        """
        if len(boxes) == 1:
            return boxes[0][1] + (boxes[0][3] - boxes[0][1]) // 2
        else:
            idx = [abs(box[0] + (box[2] - box[0]) // 2 - self.img_size[1] // 2) for box in boxes]
            idx = idx.index(min(idx))
            return boxes[idx][1] + (boxes[idx][3] - boxes[idx][1]) // 2

    def _update_voltage(self):
        """Function to update voltage of the output pole

        :return: none
        """

        voltage = int(self.total_move_distance * 10000 // 3)
        self.pi.hardware_PWM(12, 1000000, voltage)

    def _take_shot(self):
        """Function to take a shot

        :return: image
        """

        if self.img_id == 501:
            self.img_id = 1
        shot_name = os.path.join('.', 'tmpshot', str(self.img_id) + '.jpg')
        os.system('fswebcam -r '+str(self.img_size[0])+'x'+str(self.img_size[1])+' --no-banner ' + shot_name)
        self._show_message("Image " + shot_name + " saved", 6)

        shot = Image.open(shot_name)
        if shot.size[0] > shot.size[1]:
            shot = shot.transpose(Image.ROTATE_270)
        return shot

    @staticmethod
    def _show_message(message, stage):
        """Function to print out system messages.

        :param message: Content of the message
        :param stage: Stage of the message
        :return: none
        """

        stages = {-1:'[   Error   ] ',
                  0: '[  HWSetUp  ] ',
                  1: '[   First   ] ',
                  2: '[   Second  ] ',
                  3: '[   Third   ] ',
                  4: '[ Recognize ] ',
                  5: '[ WorkinRgn ] ',
                  6: '[   Others  ] '}
        print(stages[stage] + message)

    @staticmethod
    def _save_message(message, stage):
        """Function to save system messages.

        :param message: Content of the message
        :param stage: Stage of the message
        :return none
        """

        stages = {-1:'[   Error   ] ',
                  0: '[  HWSetUp  ] ',
                  1: '[   First   ] ',
                  2: '[   Second  ] ',
                  3: '[   Third   ] ',
                  4: '[ Recognize ] ',
                  5: '[ WorkinRgn ] ',
                  6: '[   Others  ] '}

        with open('./log.txt', 'a') as log_file:
            log_file.write(time.strftime("%Y-%m-%d %H:%M:%S ", time.localtime()))
            log_file.write(stages[stage] + message)
            log_file.write('\n')
        log_file.close()

    def _move(self, Dir, Time, Num):
        """Function to move the camera. Should not change this part.

        :param Dir: Direction. 1 for upward, -1 for downward
        :param Time: Speed. Better use 1.5
        :param Num: Distance. 1 for 1.25 mm
        :return:
        """

        assert Dir == 1 or Dir == -1, "Dir should be 1 or -1"
        if Num < 0:
            Dir = -1
            Num = -Num
        if self.total_move_distance + Dir * Num * 1.25 > 310:
            return False

        self.total_move_distance += Dir * Num * 1.25
        Time /= float(2000)
        Num *= 400
        StepCounter = 0
        Count = 0
        while Count <= Num:
            Count += 1
            for pin in range(0, 4):
                xpin = self.StepPins[pin]
                if self.Seq[StepCounter][pin] != 0:
                    GPIO.output(xpin, True)
                else:
                    GPIO.output(xpin, False)

            StepCounter += Dir

            if StepCounter >= self.StepCount:
                StepCounter = 0
            if StepCounter < 0:
                StepCounter = self.StepCount + Dir
            time.sleep(Time)

        return True

    def _hwsetup(self):
        """Function to set up the hardware. Should not change this part.

        :return: none
        """

        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)

        GPIO.setup(self.UpBdSwid, GPIO.IN, GPIO.PUD_UP)
        GPIO.setup(self.LoBdSwid, GPIO.IN, GPIO.PUD_UP)

        for pin in self.StepPins:
            self._show_message("Setup " + str(pin) + " pins", 0)
            GPIO.setup(pin, GPIO.OUT)
            GPIO.output(pin, False)

    def _hwinit(self):
        """Function to initialize camera to origin position. Should not change this part.

        :return: none
        """

        LoSwState = GPIO.input(self.LoBdSwid)
        while LoSwState == GPIO.HIGH:
            LoSwState = GPIO.input(self.LoBdSwid)
            self._move(-1, 1.5, 0.5)
        self.total_move_distance = 0
        self._show_message("Hardware initial done", 0)
        time.sleep(1)


# getting Raspberry Pi's Unique Serial Number
def usn_check(this_usn):
    # Extract serial from cpuinfo file
    cpuserial = "0000000000000000"
    try:
        f = open('/proc/cpuinfo', 'r')
        for line in f:
            if line[0:6] == 'Serial':
                cpuserial = line[10:26]
        f.close()
    except:
        cpuserial = "ERROR000000000"

    return this_usn == cpuserial


if __name__ == '__main__':

    if usn_check('000000004b7d1674'):
        level_tracker = LevelTracker()

        while True:
            old = level_tracker.firstStage()
            if old == 300:
                continue
            _ = level_tracker.secondStage()
    else:
        raise ValueError("Raspberry Pi's unique serial number mismatched.")
