import { useDropzone } from "react-dropzone";
import styles from "./Dropzone.module.css";

function Dropzone({ open }) {
	const { getRootProps, getInputProps, acceptedFiles } = useDropzone({});

	const files = acceptedFiles.map((file) => (
		<li key={file.path}>
			{file.path} - {file.size} bytes
		</li>
	));

	let imgFile = acceptedFiles[0];
	console.log(typeof imgFile);

	return (
		<div className={styles.container}>
			<div {...getRootProps({ className: "dropzone" })}>
				<input {...getInputProps()} />
				<p>Drag 'n' drop some files here</p>
			</div>
			<aside>
				<ul>{files}</ul>
			</aside>
		</div>
	);
}

export default Dropzone;
