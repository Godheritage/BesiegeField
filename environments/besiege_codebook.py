###################Receive from game code
code_message_map = {
    "0": "SimulationStart",
    "1": "SimulationEnd",
    "2": "MachineLoaded",
    "3": "MachineDeleted",
    "4": "LevelWin",
    "5": "MainMenuLoaded",
    "6": "SelectLevelMenuLoaded",
    "7": "BuildingSceneLoaded",
    "8": "PressKey",  # +keybinary
    "9": "ReleaseKey",  # +keybinary
    "10": "ReleaseAllKeys",
    "11": "GameLaunched",
    "12": "SendGameStatInfo",  # +game stat
    "13": "LevelFail",
}
##############################code sent to game
message_code_head_map = {
    "Load_Level:": "0", #Load unity scene
    "Load_Mission:": "1",#Load official level
    "LoadMachine:": "2",#load machine + .bsg
    "ToggleSimulate:": "3",#ToggleSimulate
    "SwitchKey:": "4",#SwitchKey,+Kaycode_T or_F
    "LoadCustomLevel:": "5",#Load costum level, + .blv
    "TriggerFail:": "6",#python trigger level fail
    "CreateThumbnail:_": "7",
}

message_code_tail_map = {
    "INITIALISER":"0",#Loading game black screen
    "TITLE SCREEN":"1",#Start Scene with an earth
    "LevelSelect":"2",#Official level selection
    "MasterSceneMultiplayer":"3"#Level editor
}