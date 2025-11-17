
"use strict";

let Load = require('./Load.js')
let GetSafetyMode = require('./GetSafetyMode.js')
let AddToLog = require('./AddToLog.js')
let IsProgramRunning = require('./IsProgramRunning.js')
let GetLoadedProgram = require('./GetLoadedProgram.js')
let GetProgramState = require('./GetProgramState.js')
let RawRequest = require('./RawRequest.js')
let IsInRemoteControl = require('./IsInRemoteControl.js')
let IsProgramSaved = require('./IsProgramSaved.js')
let GetRobotMode = require('./GetRobotMode.js')
let Popup = require('./Popup.js')

module.exports = {
  Load: Load,
  GetSafetyMode: GetSafetyMode,
  AddToLog: AddToLog,
  IsProgramRunning: IsProgramRunning,
  GetLoadedProgram: GetLoadedProgram,
  GetProgramState: GetProgramState,
  RawRequest: RawRequest,
  IsInRemoteControl: IsInRemoteControl,
  IsProgramSaved: IsProgramSaved,
  GetRobotMode: GetRobotMode,
  Popup: Popup,
};
