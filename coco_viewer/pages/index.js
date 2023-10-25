import Image from "next/image";
import { useEffect, useState } from "react";

export default function Home() {
    const [isStart, setStart] = useState(false);
    const [connected, setConnected] = useState(false);
    const [onSee, setOnSee] = useState(false);
    const [timer, setTimer] = useState();

    const [url, setUrl] = useState("");
    const [width, setWidth] = useState(400);
    const [height, setHeight] = useState(400);

    const [imgID, setImgID] = useState("");

    const second = 1000;

    const see = () => {
        const link = process.env.NEXT_PUBLIC_API_LINK || "";
        const response = fetch(`${link}/see`, {
            method: "POST",
            body: JSON.stringify({
                id: imgID || "",
            }),
            mode: "cors",
            headers: {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "application/json",
            },
        })
            .then((res) => res.json())
            .then((res) => {
                console.log(res);
            });
    };

    useEffect(() => {
        if (!onSee) return;

        const tout = setTimeout(() => {
            see();
            setOnSee(false);
            getImage();
            setTimer(4);
            // Send request to the websocket peer for the data.
        }, second * 2);

        return () => clearTimeout(tout);
    }, [onSee]);

    const getImage = () => {
        const link = process.env.NEXT_PUBLIC_API_LINK || "";
        const response = fetch(`${link}/getimg`, {
            method: "GET",
            mode: "cors",
            headers: {
                "Access-Control-Allow-Origin": "*",
                "Content-Type": "application/json",
            },
        })
            .then((res) => res.json())
            .then((res) => {
                console.log(res);
                setImgID(res.id);
                setUrl(res.url);
                setWidth(res.width);
                setHeight(res.height);
                setConnected(true);
            })
            .catch((e) => console.log(e));
    };

    useEffect(() => {
        if (timer == 0) return setOnSee(true);

        if (timer){
            const tout = setTimeout(() => {
                setTimer(timer - 1);
            }, second);
    
            return () => clearTimeout(tout);
        }
    }, [timer]);

    useEffect(() => {
        if (isStart) {
            setTimer(0)
            getImage();
        }
    }, [isStart]);

    return (
        <main className={`flex h-screen w-screen  bg-[#B5DA92]`}>
            <div className='flex flex-1 justify-center items-center'>
                {isStart ? (
                    <>
                        {connected ? (
                            <div className='flex flex-1 justify-center'>
                                {/* IMAGE VIEWER */}
                                {!onSee ? (
                                    <span className='text-6xl'>{timer}</span>
                                ) : (
                                    <Image src={url} alt='image' width={width} height={height} />
                                )}
                            </div>
                        ) : (
                            <div>Server Loading....</div>
                        )}
                    </>
                ) : (
                    <div className='text-center'>
                        <div className='mb-10 text-6xl font-extrabold'>
                            <h1>사진 만장 보기</h1>
                        </div>
                        <button
                            className='px-10 py-5 font-bold text-3xl bg-white text-gray-600 rounded-xl hover:bg-gray-200 hover:text-red-500 hover:scale-125'
                            onClick={() => setStart(true)}
                        >
                            Start
                        </button>
                    </div>
                )}
            </div>
        </main>
    );
}
