import Head from "next/head";
import axios from "axios";
import {
	ChakraProvider,
	Heading,
	Flex,
	Button,
	IconButton,
	HStack,
	VStack,
	StackDivider,
	TableContainer,
	Table,
	TableCaption,
	Thead,
	Tr,
	Th,
	Tbody,
	Td,
	Text,
	Input,
	Stack,
} from "@chakra-ui/react";
import Nav from "../components/Nav/Nav";
import UploadAndDisplayImage from "../components/UploadAndDisplayImage/UploadAndDisplayImage2";
import { useState } from "react";

const navItems = [
	{
		label: "Home",
		href: "/",
	},
	{
		label: "Classify with one modality",
		children: [
			{
				label: "Classify with Fundus",
				href: "/classify-fundus",
			},
			{
				label: "Classify with OCT",
				href: "/classify-oct",
			},
		],
	},
];

export default function Home() {
	const [image1, setImage1] = useState(null);
	const [image2, setImage2] = useState(null);
	const [ans, setAns] = useState("");
	const [pdfURLBtn, setPdfURLBtn] = useState("");

	const [patientName, setPatientName] = useState("");
	const [radiologistName, setRadiologistName] = useState("");

	const sendDataToParent = (img1, img2) => {
		setImage1(img1);
		setImage2(img2);

		console.log(`img1: ${img1}`);
		console.log(`img2: ${img2}`);
	};

	const toBase64 = (file) =>
		new Promise((resolve, reject) => {
			const reader = new FileReader();
			reader.readAsDataURL(file);
			reader.onload = () => resolve(reader.result);
			reader.onerror = (error) => reject(error);
		});

	async function sendFileToBackend() {
		let img1Data = "";
		let glaucoma_prob = "";

		var bodyFormData = new FormData();

		try {
			img1Data = await toBase64(image1);
		} catch (error) {
			console.error(error);
		}

		bodyFormData.append("image1String", img1Data);
		bodyFormData.append("patientName", patientName);
		bodyFormData.append("radiologistName", radiologistName);

		// console.log(`img2data: ${img2Data}`);

		const headers = {
			"Content-Type": "multipart/form-data",
			"Access-Control-Allow-Origin": "*",
		};

		setAns("Loading...");
		axios
			.post("http://localhost:5000/classifyFundus", bodyFormData, {
				headers: headers,
			})
			.then((response) => {
				console.log(response);
				// console.log(response.data.listItems.items);
				// console.log("Response data list items");
				// let temp = JSON.parse(response);
				// console.log(response);
				// console.log(response.data.glaucoma_prob);

				// glaucoma_prob = response.data.glaucoma_prob;
				// glaucoma_prob = glaucoma_prob * 100;

				glaucoma_prob = response.data.glaucoma_prob;

				setPdfURLBtn(
					<Button>
						<a href="http://localhost:5000/pdf" target="_blank">
							Download Report
						</a>
					</Button>
				);

				let idx = glaucoma_prob.indexOf(".");
				setAns(`Glaucoma probability: ${glaucoma_prob.slice(0, idx + 3)}%`);
			})
			.catch((error) => {
				console.log(error);
			});
	}

	return (
		<ChakraProvider>
			<div>
				<Head>
					<title>GDS</title>
					<link rel="icon" href="/favicon.ico" />
				</Head>
				<Nav NAV_ITEMS={navItems} />
				<Flex height="30vh" alignItems="center" justifyContent="center">
					<Heading as="h1" size="4xl" noOfLines={1} textColor="blue.400">
						Classify Fundus
					</Heading>
				</Flex>
				<HStack spacing={8}>
					<Flex height="20vh" width="50vw" justifyContent="center">
						<UploadAndDisplayImage
							sendDataToParent={sendDataToParent}
							sendFileToBackend={sendFileToBackend}
						/>
					</Flex>
					<Flex height="20vh" width="50vw" justifyContent="center">
						<VStack spacing={8}>
							<Flex height="60vh" width="20vw" justifyContent="center">
								<Stack spacing={3}>
									<HStack spacing={10}>
										<Text mb="8px">Patient</Text>
										<Input
											placeholder="Patient Name"
											size="md"
											onChange={(e) => setPatientName(e.target.value)}
										/>
									</HStack>
									<HStack spacing={2}>
										<Text mb="8px">Radiologist</Text>
										<Input
											placeholder="Radiologist Name"
											size="md"
											onChange={(e) => setRadiologistName(e.target.value)}
										/>
									</HStack>
								</Stack>
							</Flex>
							<Flex height="20vh" width="50vw" justifyContent="center">
								{/* <Button onClick={sendFilesToBackend}>CLASSIFY</Button> */}
							</Flex>
							<Flex height="10vh" width="50vw" justifyContent="center">
								{ans}
							</Flex>
							<Flex height="10vh" width="50vw" justifyContent="center">
								{pdfURLBtn}
							</Flex>
						</VStack>
					</Flex>
				</HStack>
			</div>
		</ChakraProvider>
	);
}
