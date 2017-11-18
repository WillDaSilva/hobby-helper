let ws = new WebSocket('ws://' + window.document.location.host);

ws.onmessage = function(message) {
    message = JSON.parse(message.data);
    // Display on screen
}

let handleSendButton = function() {
    let textField = document.getElementById('textField');
    let message = {
        data: textField.value.trim(),
        type: 'profileURL'
    };
    textField.value = '';
    ws.send(JSON.stringify(message));
};

let handleKeyPress = function(event) {
    if (event.which === 13) {
        handleSendButton();
    }
};

document.addEventListener('DOMContentLoaded', function() {
    document.getElementById('textField').addEventListener('keypress', handleKeyPress);
});
