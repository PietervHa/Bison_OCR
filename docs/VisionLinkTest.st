(*
  VisionLinkTest - PLC side of the vision PC TCP link (test program)
  Omron NJ/NX, Sysmac Studio, Structured Text

  The PLC is the TCP CLIENT, the vision PC is the server on port 5001.
  Protocol as implemented in backend/core/tcp_trigger_server.py
  with vision_mode: "ocread":

     PLC -> PC   1 byte 16#01                trigger one vision cycle
     PC  -> PLC  '<text that was read>' LF   read OK, PLC compares the text
                 'NOK' LF                    no usable read, camera or timeout

  How to test: go online, open a Watch tab, add the variables marked [W]
  and write TRUE to ManualTrigger (use Modify, NOT Force). The program
  resets it by itself, so write TRUE once per cycle.
  In production, replace ManualTrigger with a rising edge of the sensor input.

  INTERNAL VARIABLES (enter these in the program's variable table)

  Name            Data Type               Initial Value      Comment
  --------------  ----------------------  -----------------  ----------------------------------
  ManualTrigger   BOOL                    FALSE              [W] simulated sensor
  PcIp            STRING[20]              '192.168.250.10'   laptop IP, same subnet as PLC Port 1
  PcPort          UINT                    5001               trigger.port in default.yaml
  Expected        STRING[80]              ''                 [W] text you expect to be printed
  Connected       BOOL                    FALSE              [W] TCP connection to PC is up
  Response        STRING[80]              ''                 [W] raw answer line, without LF
  ResultOK        BOOL                    FALSE              [W] TRUE = text read, FALSE = NOK
  OcrText         STRING[80]              ''                 [W] text read by OCR, empty on NOK
  TextMatches     BOOL                    FALSE              [W] OcrText = Expected
  CycleCount      UDINT                   0                  [W] completed trigger/answer cycles
  State           INT                     0                  [W] state machine step
  LastErrorID     WORD                                       [W] ErrorID of last failed socket call
  LastErrorState  INT                     0                  [W] State in which it failed
  PcSocket        _sSOCKET
  SendBuf         ARRAY[0..1] OF BYTE
  RcvBuf          ARRAY[0..128] OF BYTE                      128 data bytes + 1 spare
  RcvLen          UINT                    0
  ConnectExe      BOOL                    FALSE
  SendExe         BOOL                    FALSE
  RcvExe          BOOL                    FALSE
  CloseExe        BOOL                    FALSE
  fbConnect       SktTCPConnect
  fbSend          SktTCPSend
  fbRcv           SktTCPRcv
  fbClose         SktClose
  RetryTimer      TON
*)

// ---------------- state machine ----------------
CASE State OF
    0:  // open connection to the vision PC. fbConnect must see Execute FALSE
        // for at least one scan before TRUE, or it never starts (first scan after RUN).
        Connected := FALSE;
        ConnectExe := FALSE;
        State := 2;

    2:  ConnectExe := TRUE;
        State := 1;

    1:  IF fbConnect.Done THEN
            PcSocket := fbConnect.Socket;
            ConnectExe := FALSE;
            Connected := TRUE;
            State := 10;
        ELSIF fbConnect.Error THEN
            LastErrorID := fbConnect.ErrorID;
            LastErrorState := State;
            ConnectExe := FALSE;
            State := 80;                    // PC app not running yet? wait and retry
        END_IF;

    10: // connected and idle: wait for the (simulated) sensor
        IF ManualTrigger THEN
            ManualTrigger := FALSE;         // one-shot: write TRUE again for the next cycle
            SendBuf[0] := BYTE#16#01;
            RcvLen := UINT#0;
            Response := '';
            SendExe := TRUE;
            State := 20;
        END_IF;

    20: // sending the trigger byte
        IF fbSend.Done THEN
            SendExe := FALSE;
            RcvExe := TRUE;
            State := 30;
        ELSIF fbSend.Error THEN
            LastErrorID := fbSend.ErrorID;
            LastErrorState := State;
            SendExe := FALSE;
            State := 90;
        END_IF;

    30: // waiting for the answer (OCRead can take several seconds)
        IF fbRcv.Done THEN
            RcvExe := FALSE;
            IF fbRcv.RcvSize = UINT#0 THEN
                LastErrorState := State;    // nothing arrived within TimeOut
                State := 90;
            ELSE
                RcvLen := RcvLen + fbRcv.RcvSize;
                State := 31;
            END_IF;
        ELSIF fbRcv.Error THEN
            LastErrorID := fbRcv.ErrorID;
            LastErrorState := State;
            RcvExe := FALSE;
            State := 90;
        END_IF;

    31: // TCP can deliver the line in pieces: has the LF arrived yet?
        IF RcvBuf[RcvLen - UINT#1] = BYTE#16#0A THEN
            IF RcvLen > UINT#1 THEN
                Response := AryToString(In := RcvBuf[0], Size := RcvLen - UINT#1);
            END_IF;
            ResultOK := (Response <> 'NOK') AND (Response <> '');
            IF ResultOK THEN
                OcrText := Response;
            ELSE
                OcrText := '';
            END_IF;
            TextMatches := ResultOK AND (OcrText = Expected);
            CycleCount := CycleCount + UDINT#1;
            State := 10;
        ELSIF RcvLen >= UINT#128 THEN
            LastErrorState := State;        // 128 bytes and still no LF: not our protocol
            State := 90;
        ELSE
            RcvExe := TRUE;                 // rest of the line is still on its way
            State := 30;
        END_IF;

    80: // wait before reconnecting
        IF RetryTimer.Q THEN
            State := 0;
        END_IF;

    90: // close the socket, then reconnect
        Connected := FALSE;
        CloseExe := TRUE;
        State := 91;

    91: IF fbClose.Done OR fbClose.Error THEN
            CloseExe := FALSE;
            State := 80;
        END_IF;
END_CASE;

// ---------------- instruction calls, every cycle ----------------
RetryTimer(In := (State = 80), PT := T#3s);

fbConnect(Execute := ConnectExe, SrcTcpPort := UINT#0, DstAdr := PcIp, DstTcpPort := PcPort);

fbSend(Execute := SendExe, Socket := PcSocket, SendDat := SendBuf[0], Size := UINT#1);

// Receives straight into RcvBuf at offset RcvLen. RcvLen never exceeds 128,
// and RcvBuf has 129 elements, so RcvBuf[RcvLen] is always a valid index.
// TimeOut is in 0.1 s: 200 = 20 s, longer than trigger.timeout_s (15 s) on the PC.
fbRcv(Execute := RcvExe, Socket := PcSocket, TimeOut := UINT#200,
      Size := UINT#128 - RcvLen, RcvDat := RcvBuf[RcvLen]);

fbClose(Execute := CloseExe, Socket := PcSocket);
