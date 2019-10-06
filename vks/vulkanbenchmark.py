# Copyright (C) 2019 by geehalel@gmail.com
# This code is licensed under the MIT license (MIT) (http://opensource.org/licenses/MIT)
import time

class Benchmark:
    def __init__(self):
        # private
        self._stream = None
        self._deviceProps = None
        # public
        self.active = False
        self.outputFrameTimes = False
        self.warmup = 1
        self.duration = 10
        self.frameTimes = []
        self.filename = ''
        self.runtime = 0.0
        self.frameCount = 0
    def run(self, renderFunc, deviceProps):
        self.active = True
        self._deviceProps = deviceProps
        tMeasured = 0.0
        while (tMeasured < (self.warmup * 1000)):
            tStart = time.perf_counter()
            renderFunc()
            tDiff =  time.perf_counter() - tStart
            tMeasured += tDiff
        while self.runtime < (self.duration * 1000):
            tStart = time.perf_counter()
            renderFunc()
            tDiff =  time.perf_counter() - tStart
            self.runtime += tDiff
            self.frameTimes.append(tDiff)
            self.frameCount += 1
        print("Benchmark finished")
        print("device : "), self._deviceProps.deviceName, '(driver version :', self._deviceProps.driverVersion, ')'
        print('runtime: ', (self.runtime / 1000.0))
        print('frames : ', self.frameCount)
        print('fps    : ', self.frameCount / (self.runtime / 1000.0))
    def saveResults(self):
        self._stream = open(self.filename, 'w')
        self._stream.write('device,driverversion,duration (ms),frames,fps\n')
        self._stream.write(self._deviceProps.deviceName+','+self._deviceProps.driverVersion+','+str(self.runtime)
           +','+str(self.frameCount)+','+str(self.frameCount/(self.runtime/1000.0))+'\n')
        if self.outputFrameTimes:
            self._stream.write('\nframe,ms\n')
            for i in range(len(self.frameTimes)):
                self._stream.write(str(i)+','+str(self.frameTimes[i])+'\n')
        tMin = min(self.frameTimes)
        tMax = max(self.frameTimes)
        tAvg = sum(self.frameTimes)
        self._stream.write('best     : '+str(1000.0/tMin)+' fps ('+str(tMin)+' ms)')
        self._stream.write('worst    : '+str(1000.0/tMax)+' fps ('+str(tMax)+' ms)')
        self._stream.write('avg      : '+str(1000.0/tAvg)+' fps ('+str(tAvg)+' ms)')
        self._stream.close()
