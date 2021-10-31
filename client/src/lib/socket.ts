import { writable, readable, get } from 'svelte/store'

// export const socket = new WebSocket('ws://192.168.193.42:9999')
// export const socket = new WebSocket('ws://localhost:8005/ws')

const serverIP = "localhost";
const apiPort = "8004";
const serverURL = `ws://${serverIP}:${apiPort}/ws`;

export let socket = null

function connect(){
    console.log("ESTABLISHING WEBSOCKET CONNECTION WITH ", serverURL)
    socket = new WebSocket(serverURL);
}

connect()

// preserve the socket across HMR updates
if (import.meta.hot) {
    if (import.meta.hot.data.stores) {
        socket = import.meta.hot.data.socket
    }
    import.meta.hot.accept()
    import.meta.hot.dispose(() => {
        import.meta.hot.data.socket = socket
    })
}
  
  
export const socketOpen = readable(false, (set) => {    
    set(socket.readyState === WebSocket.OPEN);
    socket.onopen = () => { 
        console.log("WEBSOCKET CONNECTED!");
        set(true)
    };

    // socket.onclose = (closeEvent) => {
    //     const reason = closeEvent.reason
    //     set(false)
    //     while (true) {
    //       console.log('SOCKET IS CLOSED. RECONNECT WILL BE ATTEMPTED IN 1 SECOND.', reason);
    //       setTimeout(function() {
    //           connect();
    //       }, 1000);

    //       if (socket.readyState === WebSocket.OPEN){
    //         break
    //       };
    //     }
    // }
})

export function messageServer(topic:string, data:any) {
    if (get(socketOpen)) {
        socket.send(JSON.stringify({ topic, data }))
    }
}