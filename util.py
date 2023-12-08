
CAMERA_SCALER = 360.0 / 2400.0

CAMERA_BUTTON_MAPPING = {
    0: 'attack',
    1: 'use',
    2: 'pickItem',
}

KEYBOARD_BUTTON_MAPPING = {
    'key.keyboard.escape': 'ESC',

    'key.keyboard.w': 'forward',
    'key.keyboard.s': 'back',
    'key.keyboard.a': 'left',
    'key.keyboard.d': 'right',

    'key.keyboard.space': 'jump',
    'key.keyboard.left.shift': 'sneak',
    'key.keyboard.left.control': 'sprint',

    'key.keyboard.1': 'hotbar.1',
    'key.keyboard.2': 'hotbar.2',
    'key.keyboard.3': 'hotbar.3',
    'key.keyboard.4': 'hotbar.4',
    'key.keyboard.5': 'hotbar.5',
    'key.keyboard.6': 'hotbar.6',
    'key.keyboard.7': 'hotbar.7',
    'key.keyboard.8': 'hotbar.8',
    'key.keyboard.9': 'hotbar.9',
    'key.keyboard.e': 'inventory',
    'key.keyboard.f': 'swapHands',
    'key.keyboard.q': 'drop',
}

MINERL_ACTION = {
    'ESC': 0,

    'camera': [0, 0],
    'attack': 0,
    'use': 0,
    'pickItem': 0,

    'forward': 0,
    'back': 0,
    'left': 0,
    'right': 0,

    'jump': 0,
    'sneak': 0,
    'sprint': 0,

    'hotbar.1': 0,
    'hotbar.2': 0,
    'hotbar.3': 0,
    'hotbar.4': 0,
    'hotbar.5': 0,
    'hotbar.6': 0,
    'hotbar.7': 0,
    'hotbar.8': 0,
    'hotbar.9': 0,

    'inventory': 0,
    'swapHands': 0,
    'drop': 0,
}

def cap(delta, mx=180, mn=-180):
    '''
    Cap a value between two other.
    '''
    if delta > mx:
        return mx
    if delta < mn:
        return mn
    return delta

def to_minerl_action(json_action):
    '''
    Convert the VPT dataset JSONL action into a MineRL action.
    '''
    is_null_action = True
    
    action = MINERL_ACTION.copy()

    if json_action is None:
        return action, is_null_action

    # Mouse movement
    dx = cap(json_action['mouse']['dx'] * CAMERA_SCALER)
    dy = cap(json_action['mouse']['dy'] * CAMERA_SCALER)
    action['camera'] = [dy, dx]
    if dx != 0 or dy != 0:
        is_null_action = False

    # Mouse actions
    for key in json_action['mouse']['buttons']:
        if key in CAMERA_BUTTON_MAPPING:
            action[CAMERA_BUTTON_MAPPING[key]] = 1
            is_null_action = False

    # Keyboard actions
    for key in json_action['keyboard']['keys']:
        if key in KEYBOARD_BUTTON_MAPPING:
            action[KEYBOARD_BUTTON_MAPPING[key]] = 1
            is_null_action = False
    
    return action, is_null_action