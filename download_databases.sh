#!/bin/bash

trap "exit" INT

PATHS='config/paths.yaml'

check_commands() {
    # Check all the required commands are available

    commands=(
        "ffmpeg"
        "wget"
        "tar"
        "unzip"
        "7z"
        "grep"
        "find"
        "wc"
        "echo"
        "mkdir"
        "basename"
        "dirname"
        "rm"
        "mv"
    )
    missing_commands=()

    for cmd in "${commands[@]}"; do
        command -v "${cmd}" >/dev/null 2>&1 || {
            missing_commands+=("${cmd}")
        }
    done

    if [ ${#missing_commands[@]} -gt 0 ]; then
        echo "The following commands required to run the script were not found: ${missing_commands[@]}"
        exit 1
    fi
}

delete () {
    echo -n "Deleting ${1}... "
    rm -rf "${1}"
    echo "done"
}

download () {
    echo "Downloading ${2}"
    wget -P "${1}" -q --show-progress -c "${2}"
}

unzip_qod () {
    echo -n "Extracting ${1}... "
    unzip -q -o "${1}" -d "${2}"
    echo "done"
    delete "${1}"
}

unzip_po () {
    echo -n "Extracting ${1}... "
    unzip -p -o "${1}" "${2}" > "${3}"
    echo "done"
    delete "${1}"
}

untar () {
    echo -n "Extracting ${1}... "
    tar -xzf "${1}" -C "${2}" --strip-components=1
    echo "done"
    delete "${1}"
}

un7z () {
    echo -n "Extracting ${1}... "
    7z x "${1}" "${2}" -o"${3}" -y > /dev/null
    echo "done"
    delete "${1}"
}

resample () {
    ffmpeg -i "${1}" -ac 1 -ar 16000 -hide_banner -loglevel error -y "${2}"
    rm -f "${1}"
}

dir_exists_not () {
    if [ -d "${1}" ]; then
        echo "Directory ${1} already exists, skipping download"
        false
    else
        true
    fi
}

file_exists_not () {
    if [ -f "${1}" ]; then
        echo "File ${1} already exists, skipping download"
        false
    else
        true
    fi
}

download_libri() {
    # 100-hour version

    LIBRI_PATH="$(grep -oP '(?<=LIBRI: ).*' $PATHS)"
    LIBRI_PATH="${LIBRI_PATH%/}"

    if dir_exists_not "${LIBRI_PATH}/train-clean-100"; then
        download "${LIBRI_PATH}" "https://www.openslr.org/resources/12/train-clean-100.tar.gz"
        untar "${LIBRI_PATH}/train-clean-100.tar.gz" "${LIBRI_PATH}"
    fi
}

download_vctk() {
    # Only 1st microphone signal is used, 2nd microphone files are deleted
    # Files are resampled at 16kHz

    VCTK_PATH="$(grep -oP '(?<=VCTK: ).*' $PATHS)"
    VCTK_PATH="${VCTK_PATH%/}"

    VCTK_RESAMPLED_PATH="${VCTK_PATH}/16kHz_mic1"
    VCTK_48KHZ_PATH=${VCTK_PATH}/wav48_silence_trimmed

    if dir_exists_not "${VCTK_RESAMPLED_PATH}" && dir_exists_not "${VCTK_48KHZ_PATH}"; then
        download "${VCTK_PATH}" "https://datashare.ed.ac.uk/download/DS_10283_3443.zip"
        unzip_qod "${VCTK_PATH}/DS_10283_3443.zip" "${VCTK_PATH}"
        unzip_qod "${VCTK_PATH}/VCTK-Corpus-0.92.zip" "${VCTK_PATH}"
    fi

    VCTK_FILES=$(find ${VCTK_48KHZ_PATH} -type f -name "*_mic1.flac")
    VCTK_FILECOUNT_SYSTEM=$(echo "${VCTK_FILES}" | wc -l)
    VCTK_FILECOUNT_TOTAL=44455
    fileindex=$((VCTK_FILECOUNT_TOTAL-VCTK_FILECOUNT_SYSTEM))
    for file in ${VCTK_FILES}; do
        fileindex=$((fileindex+1))
        percent=$((fileindex*100/VCTK_FILECOUNT_TOTAL))
        echo -ne "Resampling VCTK to 16kHz... ${fileindex}/${VCTK_FILECOUNT_TOTAL} (${percent}%)\r"
        speaker=$(basename $(dirname ${file}))
        outdir=${VCTK_RESAMPLED_PATH}/${speaker}
        outfile=${outdir}/$(basename ${file})
        mkdir -p ${outdir}
        resample ${file} ${outfile}
    done
    echo
    delete ${VCTK_48KHZ_PATH}
}

download_clarity() {
    # Files are resampled at 16kHz and converted to FLAC

    CLARITY_PATH="$(grep -oP '(?<=CLARITY: ).*' $PATHS)"
    CLARITY_PATH="${CLARITY_PATH%/}"

    CLARITY_AUDIO_PATH="${CLARITY_PATH}/audio"

    if dir_exists_not "${CLARITY_AUDIO_PATH}"; then
        download "${CLARITY_PATH}" "https://salford.figshare.com/ndownloader/files/33974840"
        untar "${CLARITY_PATH}/33974840" "${CLARITY_PATH}"
    fi

    CLARITY_FILES=$(find "${CLARITY_AUDIO_PATH}" -type f -name "*.wav")
    CLARITY_FILECOUNT_SYSTEM=$(echo "${CLARITY_FILES}" | wc -l)
    CLARITY_FILECOUNT_TOTAL=11352
    fileindex=$((CLARITY_FILECOUNT_TOTAL-CLARITY_FILECOUNT_SYSTEM))
    for file in ${CLARITY_FILES}; do
        fileindex=$((fileindex+1))
        percent=$((fileindex*100/CLARITY_FILECOUNT_TOTAL))
        echo -ne "Resampling Clarity to 16kHz... ${fileindex}/${CLARITY_FILECOUNT_TOTAL} (${percent}%)\r"
        outfile="${CLARITY_AUDIO_PATH}/$(basename ${file})"
        outfile="${outfile%.*}.flac"
        resample "${file}" "${outfile}"
    done
    echo
}

download_tau() {
    # Referred to as TAU in README.md and papers, but as DCASE in code

    TAU_PATH="$(grep -oP '(?<=DCASE: ).*' $PATHS)"
    TAU_PATH="${TAU_PATH%/}"

    TAU_AUDIO_PATH="${TAU_PATH}/audio"

    for i in {1..21}; do
        TAU_AUDIO_PATH_I="${TAU_AUDIO_PATH}_${i}"

        if dir_exists_not "${TAU_AUDIO_PATH_I}" && dir_exists_not "${TAU_AUDIO_PATH}"; then
            download "${TAU_PATH}" "https://zenodo.org/records/2589280/files/TAU-urban-acoustic-scenes-2019-development.audio.${i}.zip"
            unzip_qod "${TAU_PATH}/TAU-urban-acoustic-scenes-2019-development.audio.${i}.zip" "${TAU_PATH}"
            mv "${TAU_PATH}/TAU-urban-acoustic-scenes-2019-development/audio" "${TAU_AUDIO_PATH_I}"
        fi

        TAU_FILES=$(find "${TAU_AUDIO_PATH_I}" -type f -name "*.wav")
        TAU_FILECOUNT_SYSTEM=$(echo "${TAU_FILES}" | wc -l)
        fileindex=0
        for file in ${TAU_FILES}; do
            fileindex=$((fileindex+1))
            percent=$((fileindex*100/TAU_FILECOUNT_SYSTEM))
            echo -ne "Resampling ${TAU_AUDIO_PATH_I} to 16kHz... ${fileindex}/${TAU_FILECOUNT_SYSTEM} (${percent}%)\r"
            outfile="${TAU_AUDIO_PATH_I}/$(basename ${file})"
            outfile="${outfile%.*}.flac"
            resample "${file}" "${outfile}"
        done
        echo
    done

    if dir_exists_not "${TAU_AUDIO_PATH}"; then
        mkdir -p "${TAU_AUDIO_PATH}"
        for i in {1..21}; do
            TAU_AUDIO_PATH_I="${TAU_AUDIO_PATH}_${i}"
            echo "Moving ${TAU_AUDIO_PATH_I} to ${TAU_AUDIO_PATH}"
            mv "${TAU_AUDIO_PATH_I}"/* "${TAU_AUDIO_PATH}"
            delete "${TAU_AUDIO_PATH_I}"
        done
        delete "${TAU_PATH}/TAU-urban-acoustic-scenes-2019-development"
    fi
}

download_demand() {
    # Only channel 01 is used, channels 02-08 are deleted

    DEMAND_PATH="$(grep -oP '(?<=DEMAND: ).*' $PATHS)"
    DEMAND_PATH="${DEMAND_PATH%/}"

    DEMAND_NOISES=(
        "DKITCHEN"
        "DLIVING"
        "DWASHING"
        "NFIELD"
        "NPARK"
        "NRIVER"
        "OHALLWAY"
        "OMEETING"
        "OOFFICE"
        "PCAFETER"
        "PRESTO"
        "PSTATION"
        "SCAFE"  # not available at 16kHz
        "SPSQUARE"
        "STRAFFIC"
        "TBUS"
        "TCAR"
        "TMETRO"
    )

    for noise in "${DEMAND_NOISES[@]}"; do
        outfile="${DEMAND_PATH}/${noise}_ch01.wav"

        if [ "${noise}" == "SCAFE" ]; then
            fs="48k"
            unzip_outfile="${DEMAND_PATH}/${noise}_ch01_48k.wav"
        else
            fs="16k"
            unzip_outfile="${outfile}"
        fi

        if file_exists_not "${outfile}"; then
            download "${DEMAND_PATH}" "https://zenodo.org/records/1227121/files/${noise}_${fs}.zip"
            unzip_po "${DEMAND_PATH}/${noise}_${fs}.zip" "${noise}/ch01.wav" "${unzip_outfile}"
            if [ "${noise}" == "SCAFE" ]; then
                echo "Resampling ${unzip_outfile} to 16kHz"
                resample "${unzip_outfile}" "${outfile}"
            fi
        fi
    done
}

download_arte() {
    # The _withEQ.wav files are used

    ARTE_PATH="$(grep -oP '(?<=ARTE: ).*' $PATHS)"
    ARTE_PATH="${ARTE_PATH%/}"

    ARTE_NOISES=(
        "01_Library_binaural"
        "02_Office_binaural"
        "03_Church_1_binaural"
        "04_Living_Room_binaural"
        "05_Church_2_binaural"
        "06_Diffuse_noise_binaural"
        "07_Cafe_1_binaural"
        "08_Cafe_2_binaural"
        "09_Dinner_party_binaural"
        "10_Street_Balcony_binaural"
        "11_Train_station_binaural"  # has a different name in the archive
        "12_Food_Court_1_binaural"
        "13_Food_Court_2_binaural"
    )

    for noise in "${ARTE_NOISES[@]}"; do
        if [ "${noise}" == "11_Train_station_binaural" ]; then
            file_in_7z="11_Train_Station_binaural_withEQ.wav"
        else
            file_in_7z="${noise}_withEQ.wav"
        fi
        if file_exists_not "${ARTE_PATH}/${file_in_7z}"; then
            download "${ARTE_PATH}" "https://zenodo.org/records/3386569/files/${noise}.7z"
            un7z "${ARTE_PATH}/${noise}.7z" "${file_in_7z}" "${ARTE_PATH}"
        fi
    done
}

download_surrey () {
    # We only use rooms A to D

    SURREY_PATH="$(grep -oP '(?<=SURREY: ).*' $PATHS)"
    SURREY_PATH="${SURREY_PATH%/}"

    if file_exists_not "${SURREY_PATH}/README.md"; then
        download "${SURREY_PATH}" "https://github.com/philgzl/iosr-real-brirs-wav/archive/master.zip"
        unzip_qod "${SURREY_PATH}/master.zip" "${SURREY_PATH}"
        mv "${SURREY_PATH}/iosr-real-brirs-wav-master"/* "${SURREY_PATH}"
        delete "${SURREY_PATH}/iosr-real-brirs-wav-master"
    fi
}

download_ash () {
    ASH_PATH="$(grep -oP '(?<=ASH: ).*' $PATHS)"
    ASH_PATH="${ASH_PATH%/}"

    if file_exists_not "${ASH_PATH}/README.md"; then
        download "${ASH_PATH}" "https://github.com/ShanonPearce/ASH-IR-Dataset/archive/master.zip"
        unzip_qod "${ASH_PATH}/master.zip" "${ASH_PATH}"
        mv "${ASH_PATH}/ASH-IR-Dataset-master"/* "${ASH_PATH}"
        delete "${ASH_PATH}/ASH-IR-Dataset-master"
    fi
}

download_bras() {
    # We only use rooms CR2, CR3, CR4 and RS5

    BRAS_PATH="$(grep -oP '(?<=BRAS: ).*' $PATHS)"
    BRAS_PATH="${BRAS_PATH%/}"

    BRAS_URLS=(
        "https://depositonce.tu-berlin.de/bitstreams/53c3cf64-3547-4aa6-946b-1b4755729f2a/download"
        "https://depositonce.tu-berlin.de/bitstreams/e7b13112-0306-4596-9d9f-c6db057b0552/download"
        "https://depositonce.tu-berlin.de/bitstreams/bad0610b-293c-47cb-9926-c30c32f9b4c8/download"
        "https://depositonce.tu-berlin.de/bitstreams/ccce535a-c508-4046-8748-4458b8e73d13/download"
    )
    BRAS_PATHS_IN_ZIP=(
        "1 Scene descriptions/CR2 small room (seminar room)/BRIRs/CR2_BRIRs.sofa"
        "1 Scene descriptions/CR3 medium room (chamber music hall)/BRIRs/CR3_BRIRs.sofa"
        "1 Scene descriptions/CR4 large room (auditorium)/BRIRs/CR4_BRIRs.sofa"
        "1 Scene descriptions/RS5 diffraction (infinite wedge)/BRIRs/RS5_BRIRs.sofa"
    )

    for i in "${!BRAS_URLS[@]}"; do
        filename=$(basename "${BRAS_PATHS_IN_ZIP[$i]}")
        if file_exists_not "${BRAS_PATH}/${filename}"; then
            download "${BRAS_PATH}" "${BRAS_URLS[$i]}"
            unzip_po "${BRAS_PATH}/download" "${BRAS_PATHS_IN_ZIP[$i]}" "${BRAS_PATH}/${filename}"
        fi
    done
}

download_catt () {
    # We only use the Binaural/16k files

    CATT_PATH="$(grep -oP '(?<=CATT: ).*' $PATHS)"
    CATT_PATH="${CATT_PATH%/}"

    if dir_exists_not "${CATT_PATH}/0_0s"; then
        download "${CATT_PATH}" "https://iosr.surrey.ac.uk/software/downloads/CATT_RIRs.zip"
        unzip_qod "${CATT_PATH}/CATT_RIRs.zip" "${CATT_PATH}"
        mv "${CATT_PATH}/CATT_RIRs/Binaural/16k"/* "${CATT_PATH}"
        delete "${CATT_PATH}/CATT_RIRs"
    fi
}

check_commands
download_libri
download_vctk
download_clarity
download_tau
download_demand
download_arte
download_surrey
download_ash
download_catt
download_bras
