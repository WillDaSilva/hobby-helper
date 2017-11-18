const http = require('http');
const WebSocketServer = require('ws').Server; //provides web sockets
const ecStatic = require('ecstatic');  //provides static file server service
const url = require('url');

//static file server
let server = http.createServer(ecStatic({root: __dirname + '/client'}));

let wss = new WebSocketServer({server: server});
let redditUsernamePattern = /[\w-]{20}/;

wss.on('connection', function(ws) { //function runs when a new connection is opened
    console.log('New client connected');

    ws.on('message', function(data){
        if(!data) return;
        let parsedData = JSON.parse(data);
        if(parsedData.type == 'profileURL') {
            urlObj = url.parse(parsedData.data); //parse request type
            if(urlObj.hostname === 'www.reddit.com') {
                if(urlObj.pathname.startsWith('/u/')) {
                    redditUsername = urlObj.pathname.substring(3);
                }
                else if(urlObj.pathname.startsWith('/user/')) {
                    redditUsername = urlObj.pathname.substring(6);
                }
                if(redditUsernamePattern.test(redditUsername)) {
                    let request = http.get('https://www.reddit.com/user/' + redditUsername + '/.json', function(response){
                        let userObj = JSON.stringify(response);
                        if(!userObj.error) {
                            //Neural net processing goes here
                            //websocket return message goes here
                        }
                    });
                }
                else {
                    ws.send(JSON.stringify({type : error, error : 'Not a valid reddit username'}));
                }
            }
        }
    });
});

const PORT = 12601;
server.listen(PORT, function(error) {
    if (error) {
        console.log(error);
    }
    else {
        console.log(`Server started on port ${PORT}`);
    }
});
