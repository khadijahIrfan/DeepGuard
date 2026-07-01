console.log("SCRIPT LOADED");

const analyzeBtn = document.getElementById("analyzeBtn");

analyzeBtn.addEventListener("click", async () => {

    alert("Clicked");

    const text = document.getElementById("newsText").value;

    console.log(text);

    try {

        const response = await fetch("/predict", {

            method: "POST",

            headers: {
                "Content-Type": "application/json"
            },

            body: JSON.stringify({
                text: text
            })

        });

        console.log("Response Received");

        const data = await response.json();

        console.log(data);

        document.getElementById("predictionResult").innerText =
            data.prediction;
        document.getElementById("sentimentResult").innerText =
            data.sentiment;
        document.getElementById("metadataResult").innerText =
            "Metadata Module Connected ✅";    
    }
    catch(error)
    {
        console.log(error);

        alert("ERROR OCCURRED");
    }

});