import React, { useState } from "react";

import {
	ChakraProvider,
	Flex,
	Heading,
	Button,
	VStack,
	HStack,
} from "@chakra-ui/react";

const UploadAndDisplayImage = ({ sendDataToParent }) => {
	const [image1, setImage1] = useState(null);
	const [image2, setImage2] = useState(null);

	return (
		<ChakraProvider>
			<HStack spacing={15}>
				<VStack spacing={15}>
					<Heading as="h4" size="lg" noOfLines={1} textColor="blue.400">
						OCT
					</Heading>
					<div>
						{image1 && (
							<div>
								<img
									alt="not found"
									width={"250px"}
									src={URL.createObjectURL(image1)}
								/>
								<br />
								<Button
									colorScheme="red"
									size="xs"
									onClick={() => setImage1(null)}
								>
									Remove
								</Button>
							</div>
						)}
					</div>
					<input
						type="file"
						name="myImage"
						onChange={(event) => {
							console.log(event.target.files[0]);
							setImage1(event.target.files[0]);
							// if (image2 !== null) {
							sendDataToParent(image1, image2);
							// }
						}}
					/>
				</VStack>

				{/* -------------------------------------------------- */}

				<VStack spacing={15}>
					<Heading as="h4" size="lg" noOfLines={1} textColor="blue.400">
						Fundus
					</Heading>
					<div>
						{image2 && (
							<div>
								<img
									alt="not fount"
									width={"250px"}
									src={URL.createObjectURL(image2)}
								/>
								<br />
								<Button
									colorScheme="red"
									size="xs"
									onClick={() => setImage2(null)}
								>
									Remove
								</Button>
							</div>
						)}
					</div>
					<input
						type="file"
						name="myImage"
						onChange={(event) => {
							// console.log(event.target.files[0]);
							setImage2(event.target.files[0]);

							console.log(`image2: ${image2}`);
							// if (image1 !== null) {
							sendDataToParent(image1, event.target.files[0]);
							// }
							// sendDataToParent(event.target.files[0]);
						}}
					/>
				</VStack>
			</HStack>
		</ChakraProvider>
	);
};

export default UploadAndDisplayImage;
